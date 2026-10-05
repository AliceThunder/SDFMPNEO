from pathlib import Path
import re

import sdfmpneo


def _quoted_project_value(text, key):
    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None, f"missing project metadata field: {key}"
    return match.group(1)


def test_package_metadata_matches_current_tensor_rom_release():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert _quoted_project_value(text, "version") == sdfmpneo.__version__
    description = _quoted_project_value(text, "description").lower()
    assert "geometry-to-spatial-joule" in description
    assert "geometry-local thermal" in description
    assert "fgmres" not in description
    assert "full-space" not in description
