import yaml
from pathlib import Path

def load_cfg(name: str) -> dict:
    here = Path(__file__).resolve().parents[2]  # project root
    with open(here / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
