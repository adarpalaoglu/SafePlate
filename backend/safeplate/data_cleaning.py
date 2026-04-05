"""
SafePlate data cleaning pipeline.

Standalone script that reproduces the full cleaning pipeline from
notebooks/01_load_raw_data.ipynb.

Inputs:  data/raw/RAW_recipes.csv
Outputs: data/processed/recipes_cleaned.csv
         data/processed/recipe_ingredients.csv
         data/processed/ingredient_vocabulary.csv
         data/processed/ingredient_groups.csv
         data/processed/cleaning_metadata.json

Usage:
    python -m safeplate.data_cleaning              # skip if outputs exist
    python -m safeplate.data_cleaning --force       # regenerate everything
    python -m safeplate.data_cleaning --verbose      # print progress details
"""

import argparse
import ast
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_RECIPES_PATH = PROJECT_ROOT / "data" / "raw" / "RAW_recipes.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# Constants copied from notebook 01 cells

# Cell fbe51bd4
GENERIC_DESCRIPTORS = {
    # state / processing
    "fresh", "freshly", "frozen", "canned", "condensed", "evaporated", "prepared",
    "instant", "quick-cooking", "dried", "dry", "raw", "cooked", "smoked",
    "pickled", "refrigerated", "split",
    # cut / form
    "ground", "chopped", "diced", "minced", "sliced", "crushed",
    "shredded", "grated", "whole", "chunky", "pieces", "nuggets",
    "fillets", "steaks", "cubes", "crumbs", "crumb", "rings",
    # fat / sugar / sodium labels
    "low-fat", "reduced-fat", "fat-free", "nonfat", "light", "lite",
    "skim", "salt-free", "sugar-free", "reduced-calorie",
    # salt / butter / sweetness descriptors
    "unsalted", "salted", "sweet", "sweetened", "unsweetened",
    "dark",
    # meat / fish trim
    "lean", "extra", "boneless", "skinless", "center", "country-style", "baby",
    # size / texture / count
    "small", "medium", "large", "miniature", "little", "thin", "thick",
    "chunk", "granular", "granulated", "coarse", "fine",
    "head", "heads", "stem", "stems", "spears",
    "wedge", "wedges", "halves", "half",
    # leaves / herbs physical bits
    "leaf", "leaves", "sprig", "sprigs",
    "rind", "zest", "segments",
    # temperature
    "cold", "hot", "warm", "iced",
    # flour / cereal descriptors
    "self-rising", "self-raising", "whole wheat", "whole grain",
    "unbleached", "white", "brown", "multigrain",
    # misc adjectives
    "creamy", "plain", "sharp", "mild", "soft", "hard", "quick", "powdered",
    "ripe",
    # colour descriptors
    "black", "green", "yellow",
    # cooking states / forms
    "baking", "breasts",
    # shape/form
    "chips", "flakes",
    # generic modifiers
    "mix", "seasoning",
    # other generic / non-identity modifiers
    "bell", "sour", "heavy", "virgin", "breast", "cooking", "paste", "stock",
    "seeds", "italian",
    "boiling", "semi-sweet", "all-purpose", "jack", "sea", "spray", "dressing",
    "mapple", "cool", "half-and-half",
}

GENERIC_DESCRIPTOR_PHRASES = {
    "low sodium",
    "reduced sodium",
    "fat free",
    "sugar free",
    "reduced fat",
    "non fat",
    "whole wheat",
    "whole grain",
    "self rising",
    "self raising",
    "dark brown",
    "light brown",
}

# Cell dbac6e2f
IRREGULAR_PLURAL_INGREDIENTS = {
    "loaves": "loaf",
    "leaves": "leaf",
    "halves": "half",
    "chives": "chives",
    "greens": "greens",
}

# Cell 223defd0
ALWAYS_REMOVE_INGREDIENTS = {
    "water",
    "salt",
    "salt and pepper",
    "oil",
    "olive oil",
    "vegetable oil",
    "ice",
}

# Cell f361599a
CONNECTOR_TOKENS = {"and", "&", "'n"}

# Cell df253b06
GROUPING_STOP_TOKENS = {
    "and", "or", "of", "&", "'n",
    "with", "without", "plus",
    "in", "on", "for", "to", "from",
    "the", "a", "an",
    # container / format
    "sauce", "juice", "syrup", "soup", "broth", "dressing", "topping", "spray",
    # bready / pastry forms
    "bread", "pie", "pasta", "noodle", "roll", "cake", "crust", "shell",
    "pastry", "dough",
    # "junk drawer" / vague
    "mixed", "seasoning", "spice", "concentrate", "drink",
    # generic candy / drinks
    "candy", "cookie", "cooky", "cracker", "cereal", "soda", "liqueur", "tea",
    "coffee",
    # generic veg/fruit umbrella terms
    "vegetable", "greens", "fruit",
    "powder", "seed", "salad", "bar", "round", "liquid", "food", "bag", "stick",
    "crescent", "unbaked", "9-inch", "extract", "water",
    "french", "english", "de", "chop", "italian", "chinese", "deep", "roasted",
    "dripping", "double-crust", "mexican", "sea", "greek",
    "cube", "7-up", "essence", "mix", "roast", "sausage", "original", "bit",
    "meat", "nut", "red", "blue", "spicy", "paper", "foil",
    "wood", "wrapper", "toothpick", "bay", "top", "bottom", "dinner", "mountain",
    "paper", "cup", "wax", "wrap", "stick", "wooden",
    "toothpick", "pan", "spray", "crystal", "jar", "bottled", "slice", "puff",
    "creme", "pudding", "seasoned", "dish", "taco", "skewer",
    "bun", "toast", "smoke", "link", "whip", "whipped", "spiral", "flower",
    "cocktail", "preserve", "herb", "barbecue", "ice", "ranch",
    "tail", "cane", "chily", "half-and-half",
}

MIN_INGREDIENT_FREQUENCY = 20
MAX_INGREDIENT_FREQUENCY = 60000
MIN_INGREDIENTS_PER_RECIPE = 3


def remove_descriptors_from_ingredient(ingredient: str) -> str:
    """Remove descriptor words and phrases from a single ingredient string."""
    original = ingredient.strip()
    ing = original

    for phrase in GENERIC_DESCRIPTOR_PHRASES:
        ing = ing.replace(phrase + " ", " ")
        ing = ing.replace(" " + phrase, " ")
        if ing == phrase:
            ing = ""

    words = ing.split()
    cleaned_words = [w for w in words if w not in GENERIC_DESCRIPTORS]

    if len(cleaned_words) == 0:
        return original

    return " ".join(cleaned_words)


def remove_descriptors_from_list(ingredients_list: list) -> list:
    """Apply descriptor removal to every ingredient in the list."""
    return [remove_descriptors_from_ingredient(ing) for ing in ingredients_list]


def singularise_token(token: str) -> str:
    """Convert a single word to its singular form using minimal, safe plural rules."""
    word = token.strip().lower()
    word = word.strip(string.punctuation)

    if not word or not word.isalpha():
        return word

    if word in IRREGULAR_PLURAL_INGREDIENTS:
        return IRREGULAR_PLURAL_INGREDIENTS[word]

    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"

    if word.endswith("oes") and len(word) > 3:
        return word[:-2]

    if word.endswith("s"):
        if not word.endswith("ss") and not word.endswith("us"):
            return word[:-1]

    return word


def normalise_plurals_in_ingredient(ingredient: str) -> str:
    """Apply plural -> singular normalisation to each word inside a single ingredient string."""
    tokens = ingredient.split()
    singular_tokens = [singularise_token(t) for t in tokens]
    return " ".join(singular_tokens)


def normalise_plurals_in_list(ingredients_list: list) -> list:
    """Apply plural normalisation to each ingredient in a list."""
    return [normalise_plurals_in_ingredient(ing) for ing in ingredients_list]


def filter_by_frequency_and_background(
    ingredients_list: list,
    frequency_lookup: dict,
    min_freq: int,
    max_freq: int,
    always_remove_set: Optional[set] = None,
) -> list:
    """Remove ingredients that are too rare, too common, or background noise."""
    if always_remove_set is None:
        always_remove_set = ALWAYS_REMOVE_INGREDIENTS

    filtered = []
    for ingredient in ingredients_list:
        if ingredient in always_remove_set:
            continue
        count = frequency_lookup.get(ingredient, 0)
        if count < min_freq or count > max_freq:
            continue
        filtered.append(ingredient)
    return filtered


def deduplicate_and_sort_ingredients(ingredients_list: list) -> list:
    """Remove duplicate ingredients in a recipe and sort alphabetically."""
    return sorted(set(ingredients_list))


def split_compound_ingredient(
    ingredient: str,
    freq_lookup: dict,
    min_freq: int,
    max_freq: int,
) -> list:
    """
    Apply data-driven logic to a single ingredient that may contain a connector
    ("and", "&", "'n"). Returns a list of 0+ ingredient strings.
    """
    text = ingredient.strip()
    if not text:
        return []

    tokens = text.split()

    connector_index = None
    for i, token in enumerate(tokens):
        if token in CONNECTOR_TOKENS:
            connector_index = i
            break

    if connector_index is None:
        return [text]

    left_tokens = tokens[:connector_index]
    right_tokens = tokens[connector_index + 1:]

    left_part = " ".join(left_tokens).strip()
    right_part = " ".join(right_tokens).strip()

    if not left_part and right_part:
        return [right_part]
    if left_part and not right_part:
        return [left_part]
    if not left_part and not right_part:
        return []

    combined_count = freq_lookup.get(text, 0)
    left_count = freq_lookup.get(left_part, 0)
    right_count = freq_lookup.get(right_part, 0)

    if combined_count > left_count and combined_count > right_count:
        return [text]

    parts_to_keep = []
    if left_count > combined_count and min_freq <= left_count <= max_freq:
        parts_to_keep.append(left_part)
    if right_count > combined_count and min_freq <= right_count <= max_freq:
        parts_to_keep.append(right_part)

    if parts_to_keep:
        return parts_to_keep

    return [text]


def expand_compound_ingredients_in_list(
    ingredients_list: list,
    frequency_lookup: dict,
    min_freq: int,
    max_freq: int,
) -> list:
    """Apply the compound-splitting logic to all ingredients in a single recipe."""
    expanded = []
    for ingredient in ingredients_list:
        split_or_keep = split_compound_ingredient(
            ingredient, frequency_lookup, min_freq, max_freq,
        )
        for item in split_or_keep:
            if item:
                expanded.append(item)
    return expanded


def choose_group_token(
    ingredient: str,
    token_freqs: dict,
    stop_tokens: set,
) -> Optional[str]:
    """
    Choose a single 'head' token for an ingredient string, based on
    splitting on spaces, ignoring grouping stopwords, and picking
    the token with the highest overall frequency.
    """
    tokens = ingredient.split()
    candidates = []
    for t in tokens:
        t_clean = t.strip()
        if not t_clean:
            continue
        if not any(c.isalpha() for c in t_clean):
            continue
        if t_clean in stop_tokens:
            continue
        candidates.append(t_clean)

    if not candidates:
        return None

    return max(candidates, key=lambda x: token_freqs.get(x, 0))



def _print(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def main(force: bool = False, verbose: bool = False) -> None:
    """
    Run the full data cleaning pipeline.

    If both key outputs (recipe_ingredients.csv and ingredient_groups.csv)
    already exist and *force* is False, the pipeline is skipped entirely.
    """
    recipe_ingredients_path = OUT_DIR / "recipe_ingredients.csv"
    ingredient_groups_path = OUT_DIR / "ingredient_groups.csv"

    if (
        not force
        and recipe_ingredients_path.exists()
        and ingredient_groups_path.exists()
    ):
        print("  [SKIP] Cleaned data already exists. Use --force to regenerate.")
        return

    # Load raw data
    if not RAW_RECIPES_PATH.exists():
        raise FileNotFoundError(
            f"Raw recipes file not found: {RAW_RECIPES_PATH}\n"
            "Place RAW_recipes.csv in data/raw/ before running."
        )

    print("  Loading raw recipes ...")
    df = pd.read_csv(RAW_RECIPES_PATH)
    _print(f"  Loaded {len(df)} recipes with {df.shape[1]} columns", verbose)

    # Parse ingredient strings into lists
    df["ingredients_raw"] = df["ingredients"]

    parse_failures = 0

    def _safe_literal_eval(x):
        nonlocal parse_failures
        try:
            return ast.literal_eval(x)
        except Exception:
            parse_failures += 1
            return []

    df["ingredients"] = df["ingredients"].apply(_safe_literal_eval)
    if parse_failures > 0:
        _print(f"  Ingredient parsing failures: {parse_failures}", verbose)

    # Lowercase
    def _lowercase_list(ingredients_list):
        if not isinstance(ingredients_list, list):
            return []
        return [str(ing).lower() for ing in ingredients_list]

    df["ingredients"] = df["ingredients"].apply(_lowercase_list)

    # Step 1: remove descriptors
    _print("  Step 1: Removing descriptors ...", verbose)
    df["ingredients"] = df["ingredients"].apply(remove_descriptors_from_list)

    # Step 2: plural -> singular normalisation
    _print("  Step 2: Normalising plurals ...", verbose)
    df["ingredients"] = df["ingredients"].apply(normalise_plurals_in_list)

    # Step 3: frequency filtering
    _print("  Step 3: Filtering by frequency ...", verbose)

    all_ingredients = []
    for ingredient_list in df["ingredients"]:
        all_ingredients.extend(ingredient_list)
    ingredient_counts = Counter(all_ingredients)

    df["ingredients"] = df["ingredients"].apply(
        lambda ings: filter_by_frequency_and_background(
            ings, ingredient_counts,
            MIN_INGREDIENT_FREQUENCY, MAX_INGREDIENT_FREQUENCY,
            ALWAYS_REMOVE_INGREDIENTS,
        )
    )

    # Step 4: deduplicate + build initial vocabulary
    _print("  Step 4: Deduplicating ...", verbose)
    df["ingredients"] = df["ingredients"].apply(deduplicate_and_sort_ingredients)

    # Recompute frequencies after dedup
    final_all = []
    for ing_list in df["ingredients"]:
        final_all.extend(ing_list)
    final_ingredient_counts = Counter(final_all)

    # Step 5: remove recipes with too few ingredients
    _print("  Step 5: Removing short recipes ...", verbose)
    df["n_clean_ingredients"] = df["ingredients"].apply(len)
    df = df[df["n_clean_ingredients"] >= MIN_INGREDIENTS_PER_RECIPE].reset_index(drop=True)
    _print(f"  Recipes remaining: {len(df)}", verbose)

    # Step 6: compound ingredient handling
    _print("  Step 6: Handling compound ingredients ...", verbose)

    compound_all = []
    for ing_list in df["ingredients"]:
        compound_all.extend(ing_list)
    compound_counts = Counter(compound_all)

    df["ingredients"] = df["ingredients"].apply(
        lambda ings: expand_compound_ingredients_in_list(
            ings, compound_counts,
            MIN_INGREDIENT_FREQUENCY, MAX_INGREDIENT_FREQUENCY,
        )
    )

    df["ingredients"] = df["ingredients"].apply(deduplicate_and_sort_ingredients)

    # Rebuild final vocabulary after compound handling
    final_all = []
    for ing_list in df["ingredients"]:
        final_all.extend(ing_list)
    final_ingredient_counts = Counter(final_all)

    ingredient_vocab_df = (
        pd.DataFrame({
            "ingredient": list(final_ingredient_counts.keys()),
            "count": list(final_ingredient_counts.values()),
        })
        .sort_values(by="ingredient")
        .reset_index(drop=True)
    )
    ingredient_vocab_df["ingredient_id"] = ingredient_vocab_df.index
    ingredient_vocab_df = ingredient_vocab_df[["ingredient_id", "ingredient", "count"]]

    _print(f"  Vocabulary size: {len(ingredient_vocab_df)}", verbose)

    # Step 7: token frequencies for grouping
    _print("  Step 7: Building token frequencies for grouping ...", verbose)

    token_counts: Counter = Counter()
    for ing_list in df["ingredients"]:
        for ing in ing_list:
            for token in ing.split():
                token_counts[token] += 1

    # Step 8: ingredient grouping
    _print("  Step 8: Grouping ingredients ...", verbose)

    group_to_ingredients: dict[str, list[str]] = defaultdict(list)
    ingredient_to_group: dict[str, str] = {}

    for ingredient in ingredient_vocab_df["ingredient"]:
        head_token = choose_group_token(ingredient, token_counts, GROUPING_STOP_TOKENS)
        group_name = head_token if head_token is not None else ingredient
        group_to_ingredients[group_name].append(ingredient)
        ingredient_to_group[ingredient] = group_name

    # Save outputs
    _print("  Saving outputs ...", verbose)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RECIPE_ID_COL = "id"

    # recipe_ingredients.csv
    cols_to_save = [RECIPE_ID_COL, "ingredients", "ingredients_raw"]
    if "n_clean_ingredients" in df.columns:
        cols_to_save.append("n_clean_ingredients")
    recipe_ingredients_df = df[cols_to_save].copy()
    recipe_ingredients_df.to_csv(recipe_ingredients_path, index=False)
    print(f"  Saved: {recipe_ingredients_path} ({len(recipe_ingredients_df)} rows)")

    # recipes_cleaned.csv
    recipes_cleaned_path = OUT_DIR / "recipes_cleaned.csv"
    df.to_csv(recipes_cleaned_path, index=False)
    print(f"  Saved: {recipes_cleaned_path} ({len(df)} rows)")

    # ingredient_vocabulary.csv
    ingredient_vocab_path = OUT_DIR / "ingredient_vocabulary.csv"
    ingredient_vocab_df.to_csv(ingredient_vocab_path, index=False)
    print(f"  Saved: {ingredient_vocab_path} ({len(ingredient_vocab_df)} rows)")

    # ingredient_groups.csv
    ingredient_groups_df = pd.DataFrame(
        [{"ingredient": ing, "group_head": grp}
         for ing, grp in ingredient_to_group.items()]
    ).sort_values(by=["group_head", "ingredient"]).reset_index(drop=True)
    ingredient_groups_df.to_csv(ingredient_groups_path, index=False)
    print(f"  Saved: {ingredient_groups_path} ({len(ingredient_groups_df)} rows)")

    # cleaning_metadata.json
    metadata = {
        "min_ingredient_frequency": MIN_INGREDIENT_FREQUENCY,
        "max_ingredient_frequency": MAX_INGREDIENT_FREQUENCY,
        "min_ingredients_per_recipe": MIN_INGREDIENTS_PER_RECIPE,
        "n_recipes_final": int(len(df)),
        "vocab_size_final": int(len(ingredient_vocab_df)),
        "files_written": [
            str(recipe_ingredients_path),
            str(recipes_cleaned_path),
            str(ingredient_vocab_path),
            str(ingredient_groups_path),
        ],
    }
    metadata_path = OUT_DIR / "cleaning_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SafePlate data cleaning pipeline",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all outputs even if they already exist",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress messages",
    )
    args = parser.parse_args()
    main(force=args.force, verbose=args.verbose)
