import sysconfig
from pathlib import Path

root = Path(__file__).resolve().parent
site = Path(sysconfig.get_paths()["purelib"])
(site / "zz_boltz_sm89.pth").write_text(str(root) + "\nimport sm89_startup\n")
print(site / "zz_boltz_sm89.pth")
