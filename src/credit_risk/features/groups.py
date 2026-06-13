"""Load feature group rules from configs/feature_groups.yaml and assign group per column."""
import logging
import re
from pathlib import Path

import yaml

# credit_risk/features/groups.py -> credit_risk -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOG = logging.getLogger(__name__)


def _load_feature_groups_config() -> list:
    path = _PROJECT_ROOT / "configs" / "feature_groups.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Feature groups config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "feature_groups" not in data:
        raise ValueError("configs/feature_groups.yaml must define 'feature_groups' list")
    return data["feature_groups"]


def get_feature_group_for_column(column: str, compiled: list[tuple[str, re.Pattern]]) -> str:
    """Return the first matching group name for a column."""
    for name, pattern in compiled:
        if pattern.search(column):
            return name
    return "other"


def build_feature_group_mapping(columns: list[str]) -> dict[str, str]:
    """Build column -> feature_group using configs/feature_groups.yaml (regex-based)."""
    raw = _load_feature_groups_config()
    compiled = []
    for item in raw:
        name = item.get("name")
        pat = item.get("pattern")
        if not name or not pat:
            raise ValueError("Each feature_groups entry must have 'name' and 'pattern'")
        compiled.append((name, re.compile(pat)))
    mapping = {col: get_feature_group_for_column(col, compiled) for col in columns}
    LOG.debug("Built feature group mapping for %d columns", len(mapping))
    return mapping
