#!/usr/bin/env python3
"""Self-contained test suite for GroupAwareTriggerDetector.

Stubs out the safeplate package so the test can run standalone without
the full repo (no Flask, no data files, no other dependencies beyond numpy).
"""

import importlib.util
import math
import os
import sys
from types import ModuleType

# ───────────────────────────────────────────────────────────────────────────
# 1.  Stub out safeplate package in sys.modules BEFORE importing the module
# ───────────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    return name.strip().lower()


class StubGroupManager:
    """Minimal stand-in for IngredientGroupManager."""

    def __init__(self, groups: dict[str, list[str]] | None = None):
        self._groups = {}
        self._ing_to_group: dict[str, str] = {}
        if groups:
            for g, members in groups.items():
                g_norm = _normalize(g)
                member_norms = [_normalize(m) for m in members]
                self._groups[g_norm] = member_norms
                for m in member_norms:
                    self._ing_to_group[m] = g_norm

    def get_group(self, ingredient: str):
        return self._ing_to_group.get(_normalize(ingredient))

    def get_group_members(self, group_head: str) -> list[str]:
        return self._groups.get(_normalize(group_head), [])

    def get_all_groups(self) -> dict[str, list[str]]:
        return dict(self._groups)


_default_stub = StubGroupManager()

# Create package hierarchy
_pkg = ModuleType("safeplate")
_pkg.__path__ = []
sys.modules["safeplate"] = _pkg

_utils = ModuleType("safeplate.utils")
_utils.normalize_ingredient_name = _normalize
sys.modules["safeplate.utils"] = _utils

_groups = ModuleType("safeplate.ingredient_groups")
_groups.IngredientGroupManager = StubGroupManager
_groups.get_group_manager = lambda: _default_stub
sys.modules["safeplate.ingredient_groups"] = _groups

# Load the module under test
_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "backend", "safeplate", "group_aware_trigger_detector.py",
)
_spec = importlib.util.spec_from_file_location(
    "safeplate.group_aware_trigger_detector", _module_path
)
_mod = importlib.util.module_from_spec(_spec)
_mod.__package__ = "safeplate"
sys.modules["safeplate.group_aware_trigger_detector"] = _mod
_spec.loader.exec_module(_mod)

# Pull names into local scope
GroupAwareTriggerDetector = _mod.GroupAwareTriggerDetector
ShrinkageResult = _mod.ShrinkageResult
GroupShrinkageResult = _mod.GroupShrinkageResult
FullShrinkageOutput = _mod.FullShrinkageOutput
_binomial_survival = _mod._binomial_survival
_raw_estimates = _mod._raw_estimates

import numpy as np

# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

_failures: list[str] = []


def _find(results: list, name: str):
    """Find a ShrinkageResult by ingredient name."""
    for r in results:
        if r.ingredient == _normalize(name):
            return r
    return None


def _find_group(groups: list, name: str):
    """Find a GroupShrinkageResult by group name."""
    for g in groups:
        if g.group == _normalize(name):
            return g
    return None


def _assert(test_name: str, condition: bool, msg: str = ""):
    if not condition:
        _failures.append(f"  FAIL: {test_name}: {msg}")
        return False
    return True


# ───────────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────────

def test_1_all_triggers():
    """All group members are triggers.  One untested member."""
    gm = StubGroupManager({
        "cheese": ["ched", "mozz", "parm", "brie", "gouda", "edam"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [
        ("ched",  8, 1),
        ("mozz",  7, 2),
        ("parm",  9, 1),
        ("brie",  8, 2),
        ("gouda", 7, 1),
    ]
    out = det.run_shrinkage(triggers, all_group_members=gm.get_all_groups())

    ok = True
    for name in ["ched", "mozz", "parm", "brie", "gouda"]:
        r = _find(out.ingredients, name)
        ok &= _assert("T1", r is not None, f"{name} missing")
        ok &= _assert("T1", not r.protected, f"{name} should not be protected")
        ok &= _assert("T1", r.final_OR > 1.5,
                       f"{name} final_OR={r.final_OR:.4f}, expected > 1.5")

    edam = _find(out.ingredients, "edam")
    ok &= _assert("T1", edam is not None, "edam missing")
    ok &= _assert("T1", edam.evidence_source == "group_mean",
                   f"edam evidence_source={edam.evidence_source}")
    ok &= _assert("T1", edam.final_OR > 2.0,
                   f"edam final_OR={edam.final_OR:.4f}, expected > 2.0")

    print("TEST 1 —", "PASS" if ok else "FAIL")


def test_2_one_safe():
    """One genuinely safe member among triggers."""
    gm = StubGroupManager({
        "cheese": ["ched", "mozz", "parm", "swiss"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [
        ("ched",  8, 1),
        ("mozz",  7, 2),
        ("parm",  9, 1),
        ("swiss", 0, 10),
    ]
    out = det.run_shrinkage(triggers)

    ok = True
    swiss = _find(out.ingredients, "swiss")
    ok &= _assert("T2", swiss is not None, "swiss missing")
    ok &= _assert("T2", swiss.protected is True,
                   f"swiss.protected={swiss.protected}")
    ok &= _assert("T2", swiss.confidence == "HIGH",
                   f"swiss.confidence={swiss.confidence}")
    ok &= _assert("T2", swiss.evidence_source == "protected_safe",
                   f"swiss.evidence_source={swiss.evidence_source}")
    ok &= _assert("T2", swiss.final_OR < 1.0,
                   f"swiss.final_OR={swiss.final_OR:.4f}")

    for name in ["ched", "mozz", "parm"]:
        r = _find(out.ingredients, name)
        ok &= _assert("T2", r is not None, f"{name} missing")
        ok &= _assert("T2", not r.protected, f"{name} should not be protected")
        ok &= _assert("T2", r.final_OR > 1.0,
                       f"{name} final_OR={r.final_OR:.4f}")

    print("TEST 2 —", "PASS" if ok else "FAIL")


def test_3_safe_does_not_pollute():
    """Safe ingredient must not lower μ_g."""
    gm = StubGroupManager({
        "cheese": ["ched", "parm", "mega_safe"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [
        ("ched",      8, 1),
        ("parm",      9, 1),
        ("mega_safe", 0, 1000),
    ]
    out = det.run_shrinkage(triggers)

    ok = True
    ms = _find(out.ingredients, "mega_safe")
    ok &= _assert("T3", ms is not None, "mega_safe missing")
    ok &= _assert("T3", ms.protected is True,
                   f"mega_safe.protected={ms.protected}")

    # μ_g should be high (>0.5 on log scale ≈ OR > 1.65)
    ched = _find(out.ingredients, "ched")
    ok &= _assert("T3", ched is not None, "ched missing")
    ok &= _assert("T3", ched.mu_g > 0.5,
                   f"ched.mu_g={ched.mu_g:.4f}, expected > 0.5 "
                   "(safe ingredient should be excluded from pool)")

    print("TEST 3 —", "PASS" if ok else "FAIL")


def test_4_singleton_group():
    """Single tested ingredient in group, one untested sibling."""
    gm = StubGroupManager({
        "cheese": ["ched", "untested_cheese"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [("ched", 8, 2)]
    out = det.run_shrinkage(triggers, all_group_members=gm.get_all_groups())

    ok = True

    # Group-level checks
    cg = _find_group(out.groups, "cheese")
    ok &= _assert("T4", cg is not None, "cheese group missing")
    ok &= _assert("T4", cg.tau_sq == 0.0,
                   f"tau_sq={cg.tau_sq}, expected 0.0")

    # The tested ingredient
    ched = _find(out.ingredients, "ched")
    ok &= _assert("T4", ched is not None, "ched missing")
    theta_hat = _raw_estimates(8, 2)[0]
    ok &= _assert("T4", cg.mu_g == theta_hat,
                   f"mu_g={cg.mu_g:.6f}, expected theta_hat={theta_hat:.6f}")

    # Untested sibling
    uc = _find(out.ingredients, "untested_cheese")
    ok &= _assert("T4", uc is not None, "untested_cheese missing")
    ok &= _assert("T4", abs(uc.final_OR - ched.final_OR) < 1e-10,
                   f"untested final_OR={uc.final_OR:.10f} vs "
                   f"tested final_OR={ched.final_OR:.10f}")

    print("TEST 4 —", "PASS" if ok else "FAIL")


def test_5_tau_zero_no_kill():
    """τ² = 0 collapse must not kill confirmed triggers."""
    gm = StubGroupManager({
        "spice": ["a", "b", "c"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [
        ("a", 9, 1),
        ("b", 9, 1),
        ("c", 9, 1),
    ]
    out = det.run_shrinkage(triggers)

    ok = True
    sg = _find_group(out.groups, "spice")
    ok &= _assert("T5", sg is not None, "spice group missing")
    ok &= _assert("T5", sg.tau_sq == 0.0,
                   f"tau_sq={sg.tau_sq}, expected 0.0 (all θ̂ identical)")

    for name in ["a", "b", "c"]:
        r = _find(out.ingredients, name)
        ok &= _assert("T5", r is not None, f"{name} missing")
        ok &= _assert("T5", r.confidence == "HIGH",
                       f"{name}.confidence={r.confidence}")
        ok &= _assert("T5", r.evidence_source == "own_trigger",
                       f"{name}.evidence_source={r.evidence_source}")

    print("TEST 5 —", "PASS" if ok else "FAIL")


def test_6_binomial_direction():
    """Verify the one-sided binomial survival function."""
    ok = True

    # P(X >= 9 | n=10, p=0.5)  (should be < 0.05)
    p1 = _binomial_survival(9, 10)
    ok &= _assert("T6", p1 < 0.05,
                   f"P(X>=9|n=10)={p1:.6f}, expected < 0.05")

    # P(X >= 1 | n=10, p=0.5)  (should be > 0.9)
    p2 = _binomial_survival(1, 10)
    ok &= _assert("T6", p2 > 0.9,
                   f"P(X>=1|n=10)={p2:.6f}, expected > 0.9")

    # Verify exact value for p1
    # P(X>=9|n=10) = C(10,9)*0.5^10 + C(10,10)*0.5^10 = 11/1024
    exact = 11 / 1024
    ok &= _assert("T6", abs(p1 - exact) < 1e-15,
                   f"P(X>=9|n=10)={p1}, expected {exact}")

    print("TEST 6 —", "PASS" if ok else "FAIL")


def test_7_ungrouped():
    """Ungrouped ingredient treated as singleton."""
    gm = StubGroupManager({})  # No groups at all
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [("lonely_spice", 6, 2)]
    out = det.run_shrinkage(triggers)

    ok = True
    r = _find(out.ingredients, "lonely_spice")
    ok &= _assert("T7", r is not None, "lonely_spice missing")
    ok &= _assert("T7", r.group is None,
                   f"group={r.group}, expected None")

    theta_hat = _raw_estimates(6, 2)[0]
    ok &= _assert("T7", abs(r.final_log_OR - theta_hat) < 1e-10,
                   f"final_log_OR={r.final_log_OR:.10f}, "
                   f"expected θ̂={theta_hat:.10f}")
    ok &= _assert("T7", r.tau_sq == 0.0,
                   f"tau_sq={r.tau_sq}, expected 0.0")

    print("TEST 7 —", "PASS" if ok else "FAIL")


def test_8_notebook_wrapper():
    """run_shrinkage_from_dataframe_rows convenience method."""
    gm = StubGroupManager({
        "cheese": ["ched", "mozz"]
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    rows = [
        {"ingredient": "ched", "U_case_not_ctrl": 8, "V_ctrl_not_case": 1},
        {"ingredient": "mozz", "U_case_not_ctrl": 7, "V_ctrl_not_case": 2},
    ]
    out = det.run_shrinkage_from_dataframe_rows(rows, propagate_to_groups=True)

    ok = True
    for name in ["ched", "mozz"]:
        r = _find(out.ingredients, name)
        ok &= _assert("T8", r is not None, f"{name} missing")
        ok &= _assert("T8", not np.isnan(r.final_OR),
                       f"{name} final_OR is NaN")

    print("TEST 8 —", "PASS" if ok else "FAIL")


def test_9_serialisation():
    """to_dict() produces correct structure."""
    gm = StubGroupManager({"g": ["a", "b"]})
    det = GroupAwareTriggerDetector(group_manager=gm)
    out = det.run_shrinkage([("a", 5, 2), ("b", 3, 4)])
    d = out.to_dict()

    ok = True
    ok &= _assert("T9", isinstance(d, dict), "not a dict")
    ok &= _assert("T9", "ingredients" in d, "missing 'ingredients' key")
    ok &= _assert("T9", "groups" in d, "missing 'groups' key")
    ok &= _assert("T9", isinstance(d["ingredients"], list),
                   "ingredients is not a list")
    ok &= _assert("T9", isinstance(d["groups"], list),
                   "groups is not a list")

    if d["ingredients"]:
        ing = d["ingredients"][0]
        expected_keys = {
            "ingredient", "U", "V", "raw_log_OR", "raw_OR", "sigma_sq",
            "group", "final_log_OR", "final_OR", "weight", "mu_g", "tau_sq",
            "confidence", "evidence_source", "protected",
        }
        ok &= _assert("T9", expected_keys <= set(ing.keys()),
                       f"ingredient dict missing keys: "
                       f"{expected_keys - set(ing.keys())}")

    if d["groups"]:
        grp = d["groups"][0]
        expected_grp_keys = {
            "group", "n_tested", "n_protected", "n_in_pool",
            "mu_g", "tau_sq", "members",
        }
        ok &= _assert("T9", expected_grp_keys <= set(grp.keys()),
                       f"group dict missing keys: "
                       f"{expected_grp_keys - set(grp.keys())}")

    print("TEST 9 —", "PASS" if ok else "FAIL")


def test_10_mixed_realistic():
    """Mixed scenario: cheese (3 trigger + 1 safe + 1 untested),
    rice (1 weak), standalone."""
    gm = StubGroupManager({
        "cheese": ["ched", "mozz", "parm", "swiss", "brie"],
        "rice":   ["brown_rice", "white_rice"],
    })
    det = GroupAwareTriggerDetector(group_manager=gm)
    triggers = [
        # Cheese triggers
        ("ched",  8, 1),
        ("mozz",  7, 2),
        ("parm",  9, 1),
        # Cheese safe
        ("swiss", 0, 10),
        # Rice weak signal
        ("brown_rice", 3, 4),
        # Standalone (not in any group)
        ("ginger", 5, 1),
    ]
    out = det.run_shrinkage(triggers, all_group_members=gm.get_all_groups())

    ok = True

    # Swiss is protected
    swiss = _find(out.ingredients, "swiss")
    ok &= _assert("T10", swiss is not None, "swiss missing")
    ok &= _assert("T10", swiss.protected is True,
                   f"swiss.protected={swiss.protected}")

    # Untested cheese (brie) inherits high μ_g
    brie = _find(out.ingredients, "brie")
    ok &= _assert("T10", brie is not None, "brie missing")
    ok &= _assert("T10", brie.evidence_source == "group_mean",
                   f"brie.evidence_source={brie.evidence_source}")
    ok &= _assert("T10", brie.final_OR > 1.5,
                   f"brie.final_OR={brie.final_OR:.4f}")

    # Rice is NOT protected: P(X>=4|n=7,p=0.5) = 0.5 >= 0.05
    rice = _find(out.ingredients, "brown_rice")
    ok &= _assert("T10", rice is not None, "brown_rice missing")
    ok &= _assert("T10", not rice.protected,
                   f"brown_rice.protected={rice.protected} "
                   f"(P(X>=4|n=7)=0.5, should not be protected)")

    # Standalone has group=None
    ginger = _find(out.ingredients, "ginger")
    ok &= _assert("T10", ginger is not None, "ginger missing")
    ok &= _assert("T10", ginger.group is None,
                   f"ginger.group={ginger.group}, expected None")

    print("TEST 10 —", "PASS" if ok else "FAIL")


# ───────────────────────────────────────────────────────────────────────────
# Run all tests
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_1_all_triggers()
    test_2_one_safe()
    test_3_safe_does_not_pollute()
    test_4_singleton_group()
    test_5_tau_zero_no_kill()
    test_6_binomial_direction()
    test_7_ungrouped()
    test_8_notebook_wrapper()
    test_9_serialisation()
    test_10_mixed_realistic()

    if _failures:
        print("\n" + "=" * 60)
        print(f"FAILURES ({len(_failures)}):")
        for f in _failures:
            print(f)
        print("=" * 60)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("ALL 10 TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
