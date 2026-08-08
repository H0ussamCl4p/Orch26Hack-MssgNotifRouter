"""config.py: ROOT must resolve to the repo root after the src/ layout move."""
from router.config import ROOT


def test_root_points_at_repo_root():
    # Defect 2: ROOT = parents[2] once the package lives under src/router/.
    assert (ROOT / "pyproject.toml").exists(), (
        f"ROOT does not point at the repo root; got {ROOT}"
    )
