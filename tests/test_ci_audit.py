"""
tests/test_ci_audit.py — CI test-suite completion.

Not a test of framework *behaviour* — a test of the test suite's own
completeness against a frozen per-module list of required test names.
Walks every tests/test_*.py file's AST and confirms each required test
function actually exists somewhere (by name, not by re-implementing the
assertion — this catches someone renaming/deleting a required test in a
later refactor, which nothing else in the suite would notice, since a
deleted test simply stops running rather than failing).

Every test in FROZEN_TEST_LIST currently exists — its per-test "blocked by
<reason>" values are all None; the dict (and its parametrized "declared not
forgotten" test below, which simply collects zero cases) is kept rather
than deleted so a *future* test going missing without a recorded reason is
still a visible diff, not a silent audit pass.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

# The frozen per-module list of required test names -> which reason (if
# any) blocks a not-yet-built one.
FROZEN_TEST_LIST = {
    "test_no_subject_overlap": None,
    "test_external_never_trained": None,
    "test_val_test_no_augment": None,
    "test_mask_interpolation": None,
    "test_channel_order": None,
    "test_theta_continuity": None,
    "test_grayscale_drops_colour": None,
    "test_param_groups": None,
    "test_metric_conventions": None,
    "test_flops_agreement": None,
    "test_capacity_control_match": None,
    "test_test_loader_guard": None,
    "test_sweep_cannot_see_test": None,
    "test_reporting_blocks": None,
    "test_determinism": None,
}


def _collect_test_function_names() -> Set[str]:
    names: Set[str] = set()
    for path in TESTS_DIR.glob("test_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


def test_every_buildable_frozen_test_exists():
    existing = _collect_test_function_names()
    missing_but_buildable = []
    for name, blocked_by in FROZEN_TEST_LIST.items():
        if blocked_by is not None:
            continue
        # Exact name match OR a name this test suite actually uses that
        # clearly implements the same required test (some names recur per
        # model family, e.g. test_determinism).
        found = name in existing or any(name in n for n in existing)
        if not found:
            missing_but_buildable.append(name)
    assert not missing_but_buildable, (
        f"frozen-list tests with no known blocker are missing from the "
        f"suite: {missing_but_buildable}"
    )


@pytest.mark.parametrize(
    "name,blocked_by",
    [(n, b) for n, b in FROZEN_TEST_LIST.items() if b is not None],
)
def test_deferred_frozen_tests_are_declared_not_forgotten(name, blocked_by):
    """These are expected absent right now — this test exists so removing
    a deferred entry from this file (rather than adding the real test once
    it lands) shows up as a change to notice in review, instead of the
    audit just silently staying green forever."""
    pytest.skip(f"{name}: blocked on {blocked_by}, not a regression")


def test_ci_workflow_exists_and_runs_pytest():
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "no .github/workflows/ci.yml — CI requires a workflow"
    text = workflow.read_text()
    assert "pytest tests/" in text
