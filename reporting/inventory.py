"""
reporting/inventory.py — artefact inventory mapped to the manuscript, as
data plus an audit function — the mapping lives in one place a future
rename can't silently drift out of sync with.
"""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple


class ArtefactEntry(NamedTuple):
    manuscript_item: str
    produced_by: str  # pipeline stage(s)
    source_artefact: str  # path relative to the reports/ root


# Populated by the channel-representation study's reporting renderers
# (see reporting/tables.py, reporting/figures.py). Every source_artefact
# here is relative to the reports/ root.
ARTEFACT_INVENTORY: List[ArtefactEntry] = []


def audit_artefact_inventory(reports_root: str) -> Dict[str, object]:
    """Checks which of ``ARTEFACT_INVENTORY``'s source artefacts actually
    exist on disk under *reports_root* — a run-time completeness check
    (which manuscript items are producible *right now*), not a test of the
    mapping's correctness (that's a fixed table, verified by matching spec
    §20 directly, the same way ``tests/test_ci_audit.py`` verifies spec
    §19's test list).

    Entries with ``produced_by == "manual"`` (Fig. 1, hand-drawn) or an
    empty/glob-containing source path are reported as ``skipped`` — glob
    patterns (``attribution/*.json``) name a *family* of files, not one
    this function resolves; a caller wanting that resolved can glob
    *reports_root* directly using the same pattern.

    Returns ``{"present": [...], "missing": [...], "skipped": [...],
    "total": int}``, each list of manuscript_item strings.
    """
    present, missing, skipped = [], [], []
    for entry in ARTEFACT_INVENTORY:
        if entry.produced_by == "manual" or not entry.source_artefact or "*" in entry.source_artefact:
            skipped.append(entry.manuscript_item)
            continue
        full_path = os.path.join(reports_root, entry.source_artefact)
        if os.path.exists(full_path):
            present.append(entry.manuscript_item)
        else:
            missing.append(entry.manuscript_item)

    return {
        "present": present,
        "missing": missing,
        "skipped": skipped,
        "total": len(ARTEFACT_INVENTORY),
    }
