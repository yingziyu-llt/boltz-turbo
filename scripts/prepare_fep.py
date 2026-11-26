import argparse
import gzip
import io
import json
import math
import tarfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENFF_BASE = "https://raw.githubusercontent.com/openforcefield/protein-ligand-benchmark/main/data"
RT_KCAL_298K = 0.00198720425864083 * 298.15  # 0.5924847 kcal/mol

# Fallback PDB IDs in case target.yml omits it
DEFAULT_PDBS = {
    "tyk2": "4GIH",
    "cdk2": "1H1Q",
    "jnk1": "2GMX",
    "bace": "4DJW",
    "p38": "3FLY",
    "mcl1": "4HW3",
    "ptp1b": "2QBS",
    "thrombin": "2ZFF",
}


def http_get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "boltz-turbo/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_rcsb_sequence(pdb_id: str) -> str:
    pdb_id = pdb_id.upper()
    # Try RCSB FASTA endpoint first (simple, reliable, returns polymer chain sequences)
    fasta_text = http_get(f"https://www.rcsb.org/fasta/entry/{pdb_id}", timeout=30).decode("utf-8")
    seqs = []
    cur = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    # Pick the longest standard protein polymer chain
    protein_seqs = [s for s in seqs if len(s) >= 50 and set(s) <= set("ACDEFGHIKLMNPQRSTVWYXUOBZ")]
    assert protein_seqs, f"No protein sequence found in RCSB FASTA for {pdb_id}"
    return max(protein_seqs, key=len)


def fetch_colabfold_msa(sequence: str, out_a3m: Path, host: str = "https://api.colabfold.com") -> Path:
    """Query ColabFold MMseqs2 API once and cache a3m locally for offline Boltz-2 runs."""
    out_a3m.parent.mkdir(parents=True, exist_ok=True)
    data = urlencode({"q": f">101\n{sequence}\n", "mode": "all"}).encode("utf-8")
    req = Request(f"{host}/ticket/msa", data=data, headers={"User-Agent": "boltz-turbo/1.0"})
    ticket = json.loads(urlopen(req, timeout=60).read().decode("utf-8"))
    job_id = ticket["id"]
    print(f"[MSA] Submitted ColabFold ticket {job_id} ({len(sequence)} aa)", flush=True)

    for _ in range(120):
        status = json.loads(http_get(f"{host}/status/{job_id}", timeout=30).decode("utf-8"))
        state = status.get("status")
        if state == "COMPLETE":
            break
        if state in ("ERROR", "MAINTENANCE"):
            raise RuntimeError(f"ColabFold MSA failed: {status}")
        time.sleep(3)
    else:
        raise TimeoutError(f"Timed out waiting for ColabFold ticket {job_id}")

    archive_bytes = http_get(f"{host}/result/download/{job_id}", timeout=120)
    a3m_lines = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        for member_name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
            try:
                member = tf.getmember(member_name)
            except KeyError:
                continue
            raw = tf.extractfile(member).read().decode("utf-8", errors="ignore")
            raw = raw.replace("\x00", "")
            if not a3m_lines:
                a3m_lines.append(raw.strip())
            else:
                # Skip duplicate query record (first 2 lines) from secondary MSA file
                lines = raw.strip().splitlines()
                if len(lines) > 2 and lines[0].startswith(">"):
                    a3m_lines.append("\n".join(lines[2:]))
    assert a3m_lines, "Empty a3m archive from ColabFold"
    out_a3m.write_text("\n".join(a3m_lines) + "\n")
    n_records = sum(1 for line in out_a3m.read_text().splitlines() if line.startswith(">"))
    print(f"[MSA] Saved {out_a3m} ({n_records} sequences)", flush=True)
    return out_a3m


def convert_measurement(meas: dict) -> dict:
    val = float(meas["value"])
    unit = str(meas.get("unit", "")).strip()
    mtype = str(meas.get("type", "ki")).strip().lower()
    if unit == "uM":
        ki_uM = val
        dg = RT_KCAL_298K * math.log(ki_uM * 1e-6)
    elif unit == "nM":
        ki_uM = val * 1e-3
        dg = RT_KCAL_298K * math.log(ki_uM * 1e-6)
    elif unit == "mM":
        ki_uM = val * 1e3
        dg = RT_KCAL_298K * math.log(ki_uM * 1e-6)
    elif unit == "M":
        ki_uM = val * 1e6
        dg = RT_KCAL_298K * math.log(val)
    elif unit.lower() in ("kcal/mol", "kcal"):
        dg = val
        ki_uM = math.exp(dg / RT_KCAL_298K) * 1e6
    else:
        raise ValueError(f"Unsupported measurement unit: {meas}")
    return {
        "type": mtype,
        "raw_value": val,
        "raw_unit": unit,
        "exp_ki_uM": ki_uM,
        "exp_log10_uM": math.log10(ki_uM),
        "exp_dg_kcal": dg,
    }


def prepare_target(target: str, fetch_msa: bool, msa_path: Path | None, out_root: Path) -> None:
    target = target.lower()
    tdir = out_root / target
    inputs_dir = tdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    target_meta = yaml.safe_load(http_get(f"{OPENFF_BASE}/{target}/00_data/target.yml").decode("utf-8"))
    ligands_meta = yaml.safe_load(http_get(f"{OPENFF_BASE}/{target}/00_data/ligands.yml").decode("utf-8"))

    pdb_id = str(target_meta.get("pdb") or DEFAULT_PDBS[target]).upper()
    sequence = fetch_rcsb_sequence(pdb_id)
    print(f"[{target}] PDB={pdb_id}, sequence length={len(sequence)}, raw ligands={len(ligands_meta)}", flush=True)

    msa_field: str
    if msa_path is not None:
        msa_field = str(msa_path)
    elif fetch_msa:
        cached = tdir / "msa" / f"{target}.a3m"
        if not cached.is_file():
            fetch_colabfold_msa(sequence, cached)
        msa_field = str(cached.relative_to(REPO_ROOT)) if cached.is_relative_to(REPO_ROOT) else str(cached)
    else:
        msa_field = "empty"

    records = []
    for lig_id, entry in sorted(ligands_meta.items()):
        name = str(entry.get("name", lig_id))
        raw_smiles = str(entry["smiles"])
        mol = Chem.MolFromSmiles(raw_smiles)
        assert mol is not None, f"Invalid SMILES for {name}: {raw_smiles}"
        clean_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        meas = convert_measurement(entry["measurement"])

        payload = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": sequence, "msa": msa_field}},
                {"ligand": {"id": "B", "smiles": clean_smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        (inputs_dir / f"{name}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        records.append({
            "name": name,
            "smiles": clean_smiles,
            **meas,
        })

    dgs = [r["exp_dg_kcal"] for r in records]
    manifest = {
        "dataset": "openforcefield/protein-ligand-benchmark",
        "target": target,
        "pdb": pdb_id,
        "sequence_length": len(sequence),
        "sequence": sequence,
        "msa": msa_field,
        "num_ligands": len(records),
        "exp_dg_kcal_range": [min(dgs), max(dgs)],
        "ligands": records,
    }
    (tdir / "labels.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[{target}] Wrote {len(records)} YAMLs to {inputs_dir} "
        f"(dG range: {min(dgs):.2f} .. {max(dgs):.2f} kcal/mol, msa={msa_field})",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenFF FEP+ affinity benchmark inputs for Boltz-2.")
    parser.add_argument("--targets", nargs="+", default=["tyk2", "cdk2"],
                        help="Target names in openforcefield/protein-ligand-benchmark (e.g. tyk2 cdk2 bace jnk1).")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/fep")
    parser.add_argument("--fetch-msa", action="store_true",
                        help="Fetch & cache real MSA via ColabFold MMseqs2 API once per target.")
    parser.add_argument("--msa-path", type=Path, default=None,
                        help="Explicit path to a precomputed .a3m MSA file (single-target mode).")
    args = parser.parse_args()

    if args.msa_path is not None:
        assert len(args.targets) == 1, "--msa-path only applies when preparing one target."
    for target in args.targets:
        prepare_target(target, fetch_msa=args.fetch_msa, msa_path=args.msa_path, out_root=args.out_root)


if __name__ == "__main__":
    main()
