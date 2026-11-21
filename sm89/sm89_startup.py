import os

if os.environ.get("SM89_ACTIVE") == "1":
    from sm89_opm import install
    install()
    if os.environ.get("SM89_FULL") == "1":
        from sm89_exact import install as install_exact
        from sm89_pair import install as install_pair
        install_exact()
        install_pair()
