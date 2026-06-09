import importlib


def test_public_packages_import() -> None:
    for package_name in ("agents", "combat", "character", "configs", "rules", "training"):
        assert importlib.import_module(package_name) is not None
