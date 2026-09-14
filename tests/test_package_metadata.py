from pathlib import Path
import tomllib

import sdfmpneo


def test_package_metadata_matches_current_tensor_rom_release():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata["version"] == sdfmpneo.__version__
    description = metadata["description"].lower()
    assert "geometry-to-tensor" in description
    assert "fgmres" not in description
    assert "full-space" not in description
