#!/usr/bin/env python3
"""
Integration tests for ingredient grouping and group-aware recommendations.

Tests the interaction between:
- IngredientGroupManager (ingredient_groups.py)
- RecommendationEngine (recommendation_engine.py)
- The ingredient_groups.csv data file

These tests replicate the validation originally in notebook
05b_test_group_integration.ipynb but in a more robust script form.

The notebook was written against the naive propagation API (Phase 2), which was
entirely replaced by the DerSimonian & Laird shrinkage model (Phase 3).
This script tests the parts of the API that remain stable: group lookups
and group-aware recommendation filtering.
"""

import sys
from pathlib import Path

# Add backend to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))

from safeplate.ingredient_groups import IngredientGroupManager, get_group_manager
from safeplate.embedding_loader import EmbeddingLoader
from safeplate.recommendation_engine import RecommendationEngine


def _make_group_manager() -> IngredientGroupManager:
    """Create and load a group manager with the real data file."""
    groups_path = project_root / "data" / "processed" / "ingredient_groups.csv"
    gm = IngredientGroupManager(str(groups_path))
    gm.load()
    return gm


def _make_engine() -> RecommendationEngine:
    """Create a recommendation engine with the real embeddings."""
    embedding_path = (
        project_root
        / "data"
        / "processed"
        / "embeddings"
        / "ingredient_node2vec_iter10_UNWEIGHTED_p1_q4_w10_wl80_r80_e10.txt"
    )
    loader = EmbeddingLoader(str(embedding_path))
    return RecommendationEngine(loader)


# =====================================================================
# Tests
# =====================================================================


def test_group_manager_loading():
    """Test 1: Group manager loads and provides correct statistics."""
    print("\n" + "=" * 70)
    print("TEST 1: Group Manager Loading and Statistics")
    print("=" * 70)

    gm = _make_group_manager()
    stats = gm.get_group_statistics()

    print(f"  Total ingredients : {stats['total_ingredients']}")
    print(f"  Total groups      : {stats['total_groups']}")
    print(f"  Multi-member      : {stats['multi_member_groups']}")
    print(f"  Single-member     : {stats['single_member_groups']}")
    print(f"  Largest group     : {stats['largest_group_size']}")
    print(f"  Average group size: {stats['average_group_size']:.2f}")

    assert stats["total_ingredients"] == 2698, (
        f"Expected 2698 ingredients, got {stats['total_ingredients']}"
    )
    assert stats["total_groups"] == 993, (
        f"Expected 993 groups, got {stats['total_groups']}"
    )
    assert stats["largest_group_size"] == 105, (
        f"Expected largest group = 105, got {stats['largest_group_size']}"
    )
    assert stats["multi_member_groups"] > 0
    assert stats["single_member_groups"] > 0
    assert gm.num_ingredients == 2698
    assert gm.num_groups == 993

    print("  PASS")


def test_group_lookups():
    """Test 2: O(1) group lookup functions work correctly."""
    print("\n" + "=" * 70)
    print("TEST 2: Group Lookup Functions")
    print("=" * 70)

    gm = _make_group_manager()

    # --- get_group() ---
    print("\n  get_group():")
    assert gm.get_group("cheddar cheese") == "cheese", "cheddar cheese -> cheese"
    print("    cheddar cheese -> cheese   OK")

    assert gm.get_group("tomato") == "tomato", "tomato -> tomato"
    print("    tomato         -> tomato   OK")

    assert gm.get_group("cherry tomato") == "tomato", "cherry tomato -> tomato"
    print("    cherry tomato  -> tomato   OK")

    # Unknown ingredient returns None
    assert gm.get_group("unicorn_meat") is None, "unicorn_meat -> None"
    print("    unicorn_meat   -> None     OK")

    # --- get_group_members() ---
    print("\n  get_group_members():")
    cheese_members = gm.get_group_members("cheese")
    assert len(cheese_members) == 105, (
        f"Expected 105 cheese members, got {len(cheese_members)}"
    )
    assert "cheddar cheese" in cheese_members
    print(f"    cheese: {len(cheese_members)} members (includes cheddar cheese)  OK")

    tomato_members = gm.get_group_members("tomato")
    assert len(tomato_members) > 1, "tomato group should have multiple members"
    assert "tomato" in tomato_members
    print(f"    tomato: {len(tomato_members)} members  OK")

    # Non-existent group returns empty list
    assert gm.get_group_members("unicorn") == []
    print("    unicorn: []  OK")

    # --- are_in_same_group() ---
    print("\n  are_in_same_group():")
    assert gm.are_in_same_group("cheddar cheese", "parmesan cheese") is True
    print("    cheddar cheese & parmesan cheese: True   OK")

    assert gm.are_in_same_group("tomato", "cherry tomato") is True
    print("    tomato & cherry tomato:           True   OK")

    assert gm.are_in_same_group("tomato", "cheese") is False
    print("    tomato & cheese:                  False  OK")

    # --- expand_to_group_members() ---
    print("\n  expand_to_group_members():")
    expanded = gm.expand_to_group_members(["cheddar cheese", "tomato"])
    assert "cheddar cheese" in expanded
    assert "tomato" in expanded
    # Should include other cheese members
    assert len(expanded) > 2, (
        f"Expansion should include group members, got only {len(expanded)}"
    )
    # All cheese members should be in expanded set
    for m in cheese_members:
        assert m in expanded, f"cheese member '{m}' missing from expansion"
    print(f"    ['cheddar cheese', 'tomato'] -> {len(expanded)} ingredients  OK")

    print("  PASS")


def test_group_aware_filtering():
    """Test 3: Group-aware recommendation filtering."""
    print("\n" + "=" * 70)
    print("TEST 3: Group-Aware Filtering")
    print("=" * 70)

    gm = _make_group_manager()
    engine = _make_engine()

    query = "milk"
    individual_trigger = ["cheddar cheese"]

    # --- Regular filtering: only excludes cheddar cheese ---
    regular_results = engine.find_substitutes(
        query, triggers=individual_trigger, top_n=15
    )
    regular_names = {name for name, _ in regular_results}

    print(f"\n  Query: '{query}', trigger: {individual_trigger}")
    print(f"  Regular filtering -> {len(regular_results)} results")

    # Regular filtering should NOT exclude other cheese group members
    cheese_in_regular = [
        name for name in regular_names if gm.get_group(name) == "cheese"
    ]
    print(f"  Cheese members in regular results: {len(cheese_in_regular)}")
    for c in cheese_in_regular[:5]:
        print(f"    - {c}")

    # --- Group-aware filtering: excludes entire cheese group ---
    group_results = engine.find_substitutes_group_aware(
        query_ingredient=query,
        trigger_ingredients=individual_trigger,
        trigger_groups=["cheese"],
        top_n=15,
        group_manager=gm,
    )
    group_names = {name for name, _ in group_results}

    print(f"\n  Group-aware filtering -> {len(group_results)} results")

    # Group-aware filtering MUST NOT include any cheese group member
    cheese_in_group = [
        name for name in group_names if gm.get_group(name) == "cheese"
    ]
    assert len(cheese_in_group) == 0, (
        f"Group-aware filtering should exclude all cheese members, "
        f"but found: {cheese_in_group}"
    )
    print("  No cheese members in group-aware results  OK")

    # Both should return valid (ingredient, score) pairs with scores in [0, 1]
    for name, score in group_results:
        assert isinstance(name, str) and len(name) > 0
        assert -1.0 <= score <= 1.0, f"Score {score} out of range for {name}"

    print("  All scores are valid floats  OK")
    print("  PASS")


def test_expand_triggers():
    """Test 4: Auto-expand triggers to group members."""
    print("\n" + "=" * 70)
    print("TEST 4: Auto-Expand Triggers")
    print("=" * 70)

    gm = _make_group_manager()
    engine = _make_engine()

    triggers = ["cheddar cheese", "tomato"]

    results, excluded = engine.find_substitutes_expand_triggers(
        query_ingredient="milk",
        trigger_ingredients=triggers,
        expand_to_groups=True,
        top_n=10,
        group_manager=gm,
    )

    print(f"\n  Original triggers: {triggers}")
    print(f"  Excluded after expansion: {len(excluded)} ingredients")

    # Excluded set must include all cheese members and all tomato members
    cheese_members = set(gm.get_group_members("cheese"))
    tomato_members = set(gm.get_group_members("tomato"))

    assert cheese_members.issubset(excluded), (
        "All cheese members should be in excluded set"
    )
    assert tomato_members.issubset(excluded), (
        "All tomato members should be in excluded set"
    )
    print(f"  Cheese members in excluded: {len(cheese_members)}  OK")
    print(f"  Tomato members in excluded: {len(tomato_members)}  OK")

    # Results should not include any excluded ingredient
    result_names = {name for name, _ in results}
    overlap = result_names & excluded
    assert len(overlap) == 0, f"Results contain excluded ingredients: {overlap}"
    print("  No excluded ingredients in results  OK")

    # Compare with expand_to_groups=False
    results_no_expand, excluded_no_expand = engine.find_substitutes_expand_triggers(
        query_ingredient="milk",
        trigger_ingredients=triggers,
        expand_to_groups=False,
        top_n=10,
        group_manager=gm,
    )
    assert len(excluded_no_expand) == len(triggers), (
        f"Without expansion, excluded should equal trigger count, "
        f"got {len(excluded_no_expand)}"
    )
    print(f"  Without expansion, excluded: {len(excluded_no_expand)}  OK")

    print("  PASS")


def test_trigger_group_info():
    """Test 5: Get group metadata for triggers."""
    print("\n" + "=" * 70)
    print("TEST 5: Trigger Group Info")
    print("=" * 70)

    gm = _make_group_manager()
    engine = _make_engine()

    triggers = ["cheddar cheese", "tomato", "shrimp"]
    info = engine.get_trigger_group_info(triggers, group_manager=gm)

    print(f"\n  Queried triggers: {triggers}")

    for trigger, data in info.items():
        print(f"\n  {trigger}:")
        print(f"    group      : {data['group']}")
        print(f"    group_size : {data['group_size']}")
        print(f"    members (sample): {data['group_members'][:5]}")

    # cheddar cheese should map to cheese group
    assert info["cheddar cheese"]["group"] == "cheese"
    assert info["cheddar cheese"]["group_size"] == 105
    # group_members should NOT include cheddar cheese itself
    assert "cheddar cheese" not in info["cheddar cheese"]["group_members"]
    print("\n  cheddar cheese -> cheese group, 105 members  OK")

    # tomato should map to tomato group
    assert info["tomato"]["group"] == "tomato"
    assert info["tomato"]["group_size"] > 1
    print(f"  tomato -> tomato group, {info['tomato']['group_size']} members  OK")

    # shrimp should have a group
    assert info["shrimp"]["group"] is not None
    print(f"  shrimp -> {info['shrimp']['group']} group  OK")

    print("  PASS")


def test_regression_original_methods():
    """Test 6: Original recommendation methods still work (no breaking changes)."""
    print("\n" + "=" * 70)
    print("TEST 6: Regression - Original Methods")
    print("=" * 70)

    engine = _make_engine()

    # get_similar_ingredients
    similar = engine.get_similar_ingredients("tomato", top_n=5)
    assert len(similar) == 5, f"Expected 5 results, got {len(similar)}"
    for name, score in similar:
        assert isinstance(name, str)
        assert isinstance(score, float)
    print(f"\n  get_similar_ingredients('tomato', 5) -> {len(similar)} results  OK")
    for name, score in similar:
        print(f"    - {name}: {score:.4f}")

    # find_substitutes
    subs = engine.find_substitutes("milk", triggers=["cream"], top_n=5)
    assert len(subs) == 5, f"Expected 5 results, got {len(subs)}"
    sub_names = {name for name, _ in subs}
    assert "cream" not in sub_names, "cream should be excluded"
    print(f"\n  find_substitutes('milk', triggers=['cream'], 5) -> {len(subs)} results  OK")
    for name, score in subs:
        print(f"    - {name}: {score:.4f}")

    # get_similarity_score
    score = engine.get_similarity_score("tomato", "cherry tomato")
    assert 0.0 < score <= 1.0, f"Expected positive similarity, got {score}"
    print(f"\n  get_similarity_score('tomato', 'cherry tomato') = {score:.4f}  OK")

    # batch_similar_ingredients
    batch = engine.batch_similar_ingredients(["tomato", "milk"], top_n=3)
    assert "tomato" in batch and "milk" in batch
    assert len(batch["tomato"]) == 3
    assert len(batch["milk"]) == 3
    print(f"\n  batch_similar_ingredients(['tomato', 'milk'], 3) -> OK")

    print("  PASS")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SAFEPLATE: Group Integration Test Suite")
    print("=" * 70)

    try:
        test_group_manager_loading()
        test_group_lookups()
        test_group_aware_filtering()
        test_expand_triggers()
        test_trigger_group_info()
        test_regression_original_methods()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70 + "\n")
        sys.exit(0)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
