"""Best-Known Solutions (BKS) for the Solomon VRPTW benchmark instances.

Three tables, one per problem size:
- BKS: 100-customer instances (literature / SINTEF VRPTW benchmark values).
- BKS_25: 25-customer instances, from Table V ("Best Known" column) of
  K. Prag, M. Woolway, B. A. Jacobs, "Optimising the Vehicle Routing
  Problem with Time Windows under Standardised Metrics", ISCMI 2019,
  pp. 111-116. C202 corrected from (2, 217.7) to (2, 214.7) to match the
  verified Gurobi-exact value in results_gurobi_25.xlsx.
- BKS_50: 50-customer instances, from Bernabe Dorronsoro's CVRPTW tables
  (bernabe.dorronsoro.es/vrp). RC208 has no entry — no best-known solution
  appears to have been published for it at this size; `lookup`/`gap`
  return None for it rather than guessing.

`experiment._load_instance` renames a truncated instance with a size
suffix (e.g. "C101_25cust") so it doesn't collide with the full-size
BKS entry. `lookup` understands that suffix; a name without one falls
back to the 100-customer table, and an unknown size/instance yields None.

Suffix parsing is fragile (wrong case/separator, or a caller that never
renamed the instance silently falls back to the 100-customer table). Pass
an explicit `size=` (via `size_from_config`) to skip suffix parsing
entirely when reliability matters.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

# instance_name -> (vehicles, distance)

BKS: Dict[str, Tuple[int, float]] = {
    # C1
    "C101": (10, 828.94), "C102": (10, 828.94), "C103": (10, 828.06),
    "C104": (10, 824.78), "C105": (10, 828.94), "C106": (10, 828.94),
    "C107": (10, 828.94), "C108": (10, 828.94), "C109": (10, 828.94),
    # C2
    "C201": (3, 591.56), "C202": (3, 591.56), "C203": (3, 591.17),
    "C204": (3, 590.60), "C205": (3, 588.88), "C206": (3, 588.49),
    "C207": (3, 588.29), "C208": (3, 588.32),
    # R1
    "R101": (19, 1650.80), "R102": (17, 1486.12), "R103": (13, 1292.68),
    "R104": (9, 1007.31), "R105": (14, 1377.11), "R106": (12, 1252.03),
    "R107": (10, 1104.66), "R108": (9, 960.88), "R109": (11, 1194.73),
    "R110": (10, 1118.84), "R111": (10, 1096.72), "R112": (9, 982.14),
    # R2
    "R201": (4, 1252.37), "R202": (3, 1191.70), "R203": (3, 939.50),
    "R204": (2, 825.52), "R205": (3, 994.43), "R206": (3, 906.14),
    "R207": (2, 890.61), "R208": (2, 726.82), "R209": (3, 909.16),
    "R210": (3, 939.37), "R211": (2, 885.71),
    # RC1
    "RC101": (14, 1696.95), "RC102": (12, 1554.75), "RC103": (11, 1261.67),
    "RC104": (10, 1135.48), "RC105": (13, 1629.44), "RC106": (11, 1424.73),
    "RC107": (11, 1230.48), "RC108": (10, 1139.82),
    # RC2
    "RC201": (4, 1406.94), "RC202": (3, 1365.65), "RC203": (3, 1049.62),
    "RC204": (3, 798.46), "RC205": (4, 1297.65), "RC206": (3, 1146.32),
    "RC207": (3, 1061.14), "RC208": (3, 828.14),
}

BKS_25: Dict[str, Tuple[int, float]] = {
    # C1
    "C101": (3, 191.3), "C102": (3, 190.3), "C103": (3, 190.3),
    "C104": (3, 186.9), "C105": (3, 191.3), "C106": (3, 191.3),
    "C107": (3, 191.3), "C108": (3, 191.3), "C109": (3, 191.3),
    # C2
    "C201": (2, 214.7), "C202": (2, 214.7), "C203": (2, 214.7),
    "C204": (2, 214.7), "C205": (2, 213.1), "C206": (2, 214.7),
    "C207": (2, 214.7), "C208": (2, 214.5),
    # R1
    "R101": (8, 617.1), "R102": (7, 547.1), "R103": (5, 454.6),
    "R104": (4, 416.9), "R105": (6, 530.5), "R106": (3, 465.4),
    "R107": (4, 424.3), "R108": (4, 397.3), "R109": (5, 441.3),
    "R110": (4, 444.1), "R111": (5, 428.8), "R112": (4, 393.0),
    # R2
    "R201": (4, 463.3), "R202": (4, 410.5), "R203": (3, 391.4),
    "R204": (2, 355.0), "R205": (3, 393.0), "R206": (3, 374.4),
    "R207": (3, 361.6), "R208": (1, 328.2), "R209": (2, 370.7),
    "R210": (3, 404.6), "R211": (2, 350.9),
    # RC1
    "RC101": (4, 461.1), "RC102": (3, 351.8), "RC103": (3, 332.8),
    "RC104": (3, 306.6), "RC105": (4, 411.3), "RC106": (3, 345.5),
    "RC107": (3, 298.3), "RC108": (3, 294.5),
    # RC2
    "RC201": (3, 360.2), "RC202": (3, 338.0), "RC203": (3, 326.9),
    "RC204": (3, 299.7), "RC205": (3, 338.0), "RC206": (3, 324.0),
    "RC207": (3, 298.3), "RC208": (2, 269.1),
}

BKS_50: Dict[str, Tuple[int, float]] = {
    # C1
    "C101": (5, 362.4), "C102": (5, 361.4), "C103": (4, 361.4),
    "C104": (5, 359.0), "C105": (5, 362.4), "C106": (5, 362.4),
    "C107": (5, 362.4), "C108": (5, 362.4), "C109": (5, 362.4),
    # C2
    "C201": (3, 360.2), "C202": (3, 360.2), "C203": (3, 359.8),
    "C204": (2, 353.4), "C205": (3, 359.8), "C206": (3, 359.8),
    "C207": (3, 359.6), "C208": (2, 350.5),
    # R1
    "R101": (13, 1047.0), "R102": (12, 944.9), "R103": (9, 772.9),
    "R104": (6, 631.2), "R105": (10, 906.6), "R106": (8, 793.6),
    "R107": (7, 720.4), "R108": (6, 618.2), "R109": (8, 803.2),
    "R110": (8, 724.9), "R111": (8, 724.9), "R112": (6, 651.1),
    # R2
    "R201": (6, 800.7), "R202": (5, 712.2), "R203": (5, 606.4),
    "R204": (2, 509.5), "R205": (5, 703.3), "R206": (5, 647.0),
    "R207": (4, 584.6), "R208": (2, 487.7), "R209": (4, 600.6),
    "R210": (5, 663.4), "R211": (3, 551.3),
    # RC1
    "RC101": (9, 957.9), "RC102": (8, 844.3), "RC103": (6, 712.6),
    "RC104": (5, 546.5), "RC105": (9, 888.9), "RC106": (7, 791.9),
    "RC107": (6, 664.5), "RC108": (6, 598.1),
    # RC2 — RC208 intentionally omitted, see module docstring
    "RC201": (5, 684.8), "RC202": (5, 613.6), "RC203": (4, 555.3),
    "RC204": (3, 444.2), "RC205": (5, 631.0), "RC206": (5, 610.0),
    "RC207": (4, 558.6),
}

# Registry of all known per-size tables. Extend alongside a new BKS_<N>
# dict for further truncation sizes.
_BKS_BY_SIZE: Dict[int, Dict[str, Tuple[int, float]]] = {
    25: BKS_25,
    50: BKS_50,
    100: BKS,
}

# Decimal places each size's BKS distance is reported to; a solver's
# distance is rounded to match before a gap is computed, so noise beyond
# the table's own precision doesn't register as a spurious nonzero gap.
_ROUND_DECIMALS_BY_SIZE: Dict[int, int] = {25: 1, 50: 1, 100: 2}
_DEFAULT_ROUND_DECIMALS = 2

# Matches the size suffix experiment._load_instance appends to a truncated
# instance's name, e.g. "C101_25cust" -> base "C101", size 25.
_SIZE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<size>\d+)CUST$")

_DEFAULT_CONFIG_SIZE_KEYS = ("max_customers", "n_customers", "num_customers", "size")


def size_from_config(config: Any,
                      keys: Tuple[str, ...] = _DEFAULT_CONFIG_SIZE_KEYS) -> int:
    """Read the instance size out of a run config (dict-like or attribute-
    based), trying each key in order. Raises KeyError if none resolve, or
    ValueError if a found value isn't an int — both fail loudly rather than
    silently defaulting to 100.

    >>> size = size_from_config(cfg)  # reads cfg.max_customers
    >>> gap("C101", vehicles=3, distance=192.1, size=size)
    """
    for key in keys:
        if hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                return _coerce_size(key, value)
            continue
        if isinstance(config, Mapping) and key in config:
            value = config[key]
            if value is not None:
                return _coerce_size(key, value)
            continue
    raise KeyError(
        f"None of {keys} found (or all were None) on config {config!r}; "
        f"pass the instance size explicitly or add one of those keys/"
        f"attributes to your config."
    )


def _coerce_size(key: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config key/attribute '{key}' = {value!r} is not a valid "
            f"instance size (int)."
        ) from exc


def _resolve(instance_name: str,
             size: Optional[int] = None) -> Tuple[Optional[Tuple[int, float]], int]:
    """(bks_entry, round_decimals) for instance_name. An explicit size
    bypasses suffix parsing entirely (the reliable path); otherwise parses
    the "_<N>CUST" suffix, falling back to the 100-customer table if none
    is found."""
    if size is not None:
        table = _BKS_BY_SIZE.get(size)
        decimals = _ROUND_DECIMALS_BY_SIZE.get(size, _DEFAULT_ROUND_DECIMALS)
        base = instance_name.upper()
        match = _SIZE_SUFFIX_RE.match(base)
        if match:
            base = match.group("base")
        if table is None:
            return None, decimals
        return table.get(base), decimals

    name = instance_name.upper()
    match = _SIZE_SUFFIX_RE.match(name)
    if match:
        parsed_size = int(match.group("size"))
        table = _BKS_BY_SIZE.get(parsed_size)
        decimals = _ROUND_DECIMALS_BY_SIZE.get(parsed_size, _DEFAULT_ROUND_DECIMALS)
        if table is None:
            return None, decimals
        return table.get(match.group("base")), decimals
    return BKS.get(name), _ROUND_DECIMALS_BY_SIZE.get(100, _DEFAULT_ROUND_DECIMALS)


def lookup(instance_name: str, size: Optional[int] = None) -> Optional[Tuple[int, float]]:
    """BKS for instance_name. Pass size explicitly (see size_from_config) for
    reliable resolution regardless of naming; otherwise resolves via a
    truncation-size suffix if present, else the 100-customer table. None if
    unknown."""
    bk, _decimals = _resolve(instance_name, size=size)
    return bk


def gap(instance_name: str, vehicles: int, distance: float,
        size: Optional[int] = None) -> Dict[str, Optional[float]]:
    """Percentage gaps to BKS for vehicles and distance (or None if unknown).
    `distance` is rounded to the matched table's own precision first, so
    finer-grained rounding doesn't register as a spurious gap."""
    bk, decimals = _resolve(instance_name, size=size)
    if bk is None:
        return {"bks_vehicles": None, "bks_distance": None,
                "gap_vehicles_pct": None, "gap_distance_pct": None}
    bk_v, bk_d = bk
    distance = round(distance, decimals)
    return {
        "bks_vehicles": bk_v,
        "bks_distance": bk_d,
        "gap_vehicles_pct": round(100.0 * (vehicles - bk_v) / bk_v, 2) if bk_v else None,
        "gap_distance_pct": round(100.0 * (distance - bk_d) / bk_d, 2) if bk_d else None,
    }
