"""Generate train and three test datasets for OGBench Move Cube.

The in-distribution test set repeats every training combination for one fresh
episode. One held-out test set uses novel combinations of otherwise seen factor
levels, while the other holds out atomic feature levels such as colors, cube
sizes, or spatial regions. Exact categorical assignments are generated
deterministically from the JSON config; continuous positions are resampled
inside their regions per episode.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

DEFAULT_SPLIT_CONFIG = (
    ROOT / "configs" / "move_cube_compositional_regions.json"
)
DEFAULT_TASK_VARIATIONS = (
    "cube.start_position",
    "cube.start_yaw",
    "cube.goal_position",
    "cube.goal_yaw",
)
SPLIT_SEED_OFFSETS = {
    "train": 0,
    "test_combinations": 1_000_000,
    "test_features": 2_000_000,
    "test_indistribution": 3_000_000,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Generate compositional Move Cube datasets"
    )
    parser.add_argument("--image_shape", type=positive_int, default=224)
    parser.add_argument(
        "--env_type",
        choices=["single", "double", "triple", "quadruple", "octuple"],
        default="single",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        required=True,
        help="Total number of training episodes (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--test_episodes",
        type=positive_int,
        default=256,
        help=(
            "Fallback total for each test dataset. Per-split totals can be set "
            "with --combination_test_episodes and --feature_test_episodes."
        ),
    )
    parser.add_argument(
        "--combination_test_episodes",
        type=positive_int,
        default=None,
        help="Total episodes in test_combinations (defaults to --test_episodes).",
    )
    parser.add_argument(
        "--feature_test_episodes",
        type=positive_int,
        default=None,
        help="Total episodes in test_features (defaults to --test_episodes).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_name", default="move-cube")
    parser.add_argument(
        "--split_config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
        help="JSON file defining factor levels and explicit train/test combinations.",
    )
    parser.add_argument(
        "--variations",
        "--extra_variations",
        dest="extra_variations",
        nargs="+",
        default=[],
        help=(
            "Additional factors to randomize. Controlled factors and the task's "
            "start/goal variations are included automatically. Avoid 'all' in "
            "the main compositional benchmark because it adds uncontrolled shifts."
        ),
    )
    parser.add_argument(
        "--physics_timestep",
        type=float,
        default=0.002,
        help="MuJoCo integration step in seconds.",
    )
    parser.add_argument(
        "--control_timestep",
        type=float,
        default=0.05,
        help="Seconds between recorded actions/frames (0.05 = 20 Hz).",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=positive_int,
        default=200,
    )
    parser.add_argument(
        "--policy_type",
        choices=["markov_oracle", "plan_oracle"],
        default="markov_oracle",
    )
    parser.add_argument("--action_noise", type=nonnegative_float, default=0.1)
    parser.add_argument(
        "--p_random_action",
        type=probability,
        default=0.1,
        help="Probability of replacing the oracle action with a random action.",
    )
    parser.add_argument("--noise_smoothing", type=probability, default=0.5)
    parser.add_argument("--min_norm", type=nonnegative_float, default=0.4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing data for this task in train/, "
            "test_indistribution/, test_combinations/, and test_features/."
        ),
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable Stable-WorldModel's per-combination progress bars.",
    )
    return parser.parse_args(argv)


def validate_timesteps(
    physics_timestep: float, control_timestep: float
) -> int:
    if physics_timestep <= 0.0 or control_timestep <= 0.0:
        raise ValueError("physics_timestep and control_timestep must be positive")
    if control_timestep < physics_timestep:
        raise ValueError(
            "control_timestep must be at least as large as physics_timestep"
        )

    ratio = control_timestep / physics_timestep
    physics_steps = round(ratio)
    if not math.isclose(ratio, physics_steps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "control_timestep must be an integer multiple of physics_timestep; "
            f"got {control_timestep} / {physics_timestep} = {ratio}"
        )
    return physics_steps


def _load_split_config(path: Path, env_type: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    supported_env_types = config.get("env_types")
    if supported_env_types and env_type not in supported_env_types:
        raise ValueError(
            f"{path} supports env_type values {supported_env_types}, "
            f"not {env_type!r}"
        )

    factors = config.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("The split config needs a non-empty 'factors' list")

    factor_names: list[str] = []
    valid_levels: dict[str, set[str]] = {}
    test_only_levels: dict[str, set[str]] = {}
    position_variations: set[str] = set()
    for factor in factors:
        name = factor.get("name")
        levels = factor.get("levels")
        if not isinstance(name, str) or not name:
            raise ValueError("Every factor needs a non-empty string 'name'")
        if name in factor_names:
            raise ValueError(f"Duplicate factor name: {name}")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(f"Factor {name!r} needs at least two levels")

        level_names = [level.get("name") for level in levels]
        if any(not isinstance(level_name, str) for level_name in level_names):
            raise ValueError(f"Every level of {name!r} needs a string 'name'")
        if len(set(level_names)) != len(level_names):
            raise ValueError(f"Factor {name!r} has duplicate level names")

        factor_kind = factor.get("kind", "variation")
        if factor_kind not in {"variation", "position_region"}:
            raise ValueError(
                f"Factor {name!r} has invalid kind {factor_kind!r}"
            )
        if factor_kind == "variation":
            if any("value" not in level for level in levels):
                raise ValueError(f"Every level of {name!r} needs a 'value'")
        else:
            variation = factor.get("variation")
            if variation not in {
                "cube.start_position",
                "cube.goal_position",
            }:
                raise ValueError(
                    f"Position-region factor {name!r} must target "
                    "'cube.start_position' or 'cube.goal_position'"
                )
            if variation in position_variations:
                raise ValueError(
                    f"Multiple position-region factors target {variation!r}"
                )
            position_variations.add(variation)
            for level in levels:
                low = level.get("low")
                high = level.get("high")
                if (
                    not isinstance(low, list)
                    or not isinstance(high, list)
                    or len(low) != 2
                    or len(high) != 2
                    or not all(
                        isinstance(value, (int, float))
                        for value in low + high
                    )
                    or any(lower >= upper for lower, upper in zip(low, high))
                ):
                    raise ValueError(
                        f"Every level of position-region factor {name!r} "
                        "needs two-dimensional numeric 'low' and 'high' bounds "
                        "with low < high"
                    )
        invalid_availability = {
            level.get("availability", "shared") for level in levels
        } - {"shared", "test_only"}
        if invalid_availability:
            raise ValueError(
                f"Factor {name!r} has invalid availability values "
                f"{sorted(invalid_availability)}"
            )
        factor_names.append(name)
        valid_levels[name] = set(level_names)
        test_only_levels[name] = {
            level["name"]
            for level in levels
            if level.get("availability", "shared") == "test_only"
        }

    split_config = config.get("splits")
    required_splits = {"train", "test_combinations", "test_features"}
    if not isinstance(split_config, dict) or set(split_config) != required_splits:
        raise ValueError(
            f"'splits' must contain exactly {sorted(required_splits)}"
        )
    for split_name, specification in split_config.items():
        count = specification.get("num_combinations")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"{split_name}.num_combinations must be a positive integer"
            )

    combination_holdout = split_config["test_combinations"].get("holdout")
    if (
        not isinstance(combination_holdout, dict)
        or combination_holdout.get("type") != "combination"
    ):
        raise ValueError(
            "test_combinations holdout.type must be 'combination'"
        )
    factor_pairs = combination_holdout.get("factor_pairs")
    if factor_pairs is not None:
        pair_factors = factor_pairs.get("factors")
        pair_values = factor_pairs.get("values")
        if (
            not isinstance(pair_factors, list)
            or len(pair_factors) != 2
            or len(set(pair_factors)) != 2
            or any(name not in valid_levels for name in pair_factors)
        ):
            raise ValueError(
                "test_combinations.holdout.factor_pairs.factors must contain "
                "two distinct factor names"
            )
        if (
            not isinstance(pair_values, list)
            or not pair_values
            or any(
                not isinstance(pair, list) or len(pair) != 2
                for pair in pair_values
            )
            or len({tuple(pair) for pair in pair_values}) != len(pair_values)
        ):
            raise ValueError(
                "test_combinations.holdout.factor_pairs.values must contain "
                "unique two-element level pairs"
            )
        for pair in pair_values:
            for factor_name, level_name in zip(pair_factors, pair):
                if (
                    level_name not in valid_levels[factor_name]
                    or level_name in test_only_levels[factor_name]
                ):
                    raise ValueError(
                        "Combination-held-out pairs must use shared levels; "
                        f"got {factor_name}={level_name!r}"
                    )
        if split_config["test_combinations"]["num_combinations"] < len(
            pair_values
        ):
            raise ValueError(
                "test_combinations.num_combinations must cover every held-out pair"
            )

    feature_holdout = split_config["test_features"].get("holdout")
    if (
        not isinstance(feature_holdout, dict)
        or feature_holdout.get("type") != "factor_level"
    ):
        raise ValueError("test_features holdout.type must be 'factor_level'")
    groups = feature_holdout.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("test_features holdout needs non-empty 'groups'")

    declared_test_only: dict[str, set[str]] = {
        name: set() for name in factor_names
    }
    group_total = 0
    for group_index, group in enumerate(groups):
        factor_name = group.get("factor")
        level_names = group.get("levels")
        count = group.get("num_combinations")
        if factor_name not in valid_levels:
            raise ValueError(
                f"Feature holdout group {group_index} has unknown factor "
                f"{factor_name!r}"
            )
        if (
            not isinstance(level_names, list)
            or not level_names
            or not set(level_names) <= test_only_levels[factor_name]
        ):
            raise ValueError(
                f"Feature holdout group {group_index} must reference test-only "
                f"levels of {factor_name!r}"
            )
        if not isinstance(count, int) or count < len(level_names):
            raise ValueError(
                f"Feature holdout group {group_index}.num_combinations must "
                "cover every listed level"
            )
        overlap = declared_test_only[factor_name] & set(level_names)
        if overlap:
            raise ValueError(
                f"Test-only levels declared in multiple groups: {sorted(overlap)}"
            )
        declared_test_only[factor_name].update(level_names)
        group_total += count

    expected_group_total = split_config["test_features"]["num_combinations"]
    if group_total != expected_group_total:
        raise ValueError(
            "Feature holdout group counts must sum to "
            f"{expected_group_total}, got {group_total}"
        )
    if declared_test_only != test_only_levels:
        raise ValueError(
            "Every level marked availability='test_only' must occur in exactly "
            "one feature-holdout group"
        )
    if not isinstance(config.get("combination_seed"), int):
        raise ValueError("'combination_seed' must be an integer")

    position_sampling = config.get("position_sampling", {})
    if not isinstance(position_sampling, dict):
        raise ValueError("'position_sampling' must be an object")
    minimum_distance = position_sampling.get(
        "minimum_start_goal_distance", 0.0
    )
    max_attempts = position_sampling.get("max_attempts", 100)
    if not isinstance(minimum_distance, (int, float)) or minimum_distance < 0:
        raise ValueError(
            "position_sampling.minimum_start_goal_distance must be non-negative"
        )
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("position_sampling.max_attempts must be positive")

    return config


def _covers_all_levels(
    combinations: list[dict[str, str]],
    required_levels: dict[str, set[str]],
) -> bool:
    return all(
        {combination[factor] for combination in combinations} == levels
        for factor, levels in required_levels.items()
    )


def _has_balanced_level_counts(
    combinations: list[dict[str, str]],
    required_levels: dict[str, set[str]],
    tolerance: int,
) -> bool:
    for factor, levels in required_levels.items():
        counts = [
            sum(combination[factor] == level for combination in combinations)
            for level in levels
        ]
        if max(counts) - min(counts) > tolerance:
            return False
    return True


def _sample_with_coverage(
    candidates: list[dict[str, str]],
    count: int,
    required_levels: dict[str, set[str]],
    seed: int,
    tolerance: int = 1,
) -> list[dict[str, str]]:
    if count > len(candidates):
        raise ValueError(
            f"Cannot choose {count} combinations from a pool of {len(candidates)}"
        )
    if any(count < len(levels) for levels in required_levels.values()):
        raise ValueError(
            f"Cannot cover every required level with only {count} combinations"
        )

    factor_names = list(required_levels)
    candidate_keys = {
        tuple(candidate[name] for name in factor_names): candidate
        for candidate in candidates
    }
    for attempt in range(10_000):
        rng = random.Random(seed + attempt)
        schedules: dict[str, list[str]] = {}
        for factor_name, required in required_levels.items():
            levels = sorted(required)
            base, remainder = divmod(count, len(levels))
            schedule = levels * base + rng.sample(levels, remainder)
            rng.shuffle(schedule)
            schedules[factor_name] = schedule

        keys = [
            tuple(schedules[name][index] for name in factor_names)
            for index in range(count)
        ]
        if len(set(keys)) != count or any(
            key not in candidate_keys for key in keys
        ):
            continue
        selected = [dict(candidate_keys[key]) for key in keys]
        if _covers_all_levels(
            selected, required_levels
        ) and _has_balanced_level_counts(
            selected, required_levels, tolerance=tolerance
        ):
            return selected

    # A nearly saturated candidate pool can make independently constructed
    # balanced schedules collide often. Fall back to subset sampling when the
    # caller allows a wider tolerance (for example the 60/256 legacy split).
    for attempt in range(10_000):
        shuffled = list(candidates)
        random.Random(seed + 10_000 + attempt).shuffle(shuffled)
        selected = shuffled[:count]
        if _covers_all_levels(
            selected, required_levels
        ) and _has_balanced_level_counts(
            selected, required_levels, tolerance=tolerance
        ):
            return selected
    raise ValueError(
        f"Could not select {count} combinations with the requested level coverage"
    )


def _factor_product(
    factor_names: list[str],
    levels: dict[str, list[str]],
) -> list[dict[str, str]]:
    return [
        dict(zip(factor_names, values))
        for values in itertools.product(*(levels[name] for name in factor_names))
    ]


def _sample_with_pair_coverage(
    *,
    candidates: list[dict[str, str]],
    count: int,
    required_levels: dict[str, set[str]],
    pair_factors: list[str],
    pairs: list[list[str]],
    seed: int,
) -> list[dict[str, str]]:
    """Build a balanced sample constrained to categorical factor pairs."""
    factor_names = list(required_levels)
    candidate_keys = {
        tuple(candidate[name] for name in factor_names): candidate
        for candidate in candidates
    }
    pair_tuples = [tuple(pair) for pair in pairs]
    for attempt in range(10_000):
        rng = random.Random(seed + attempt)
        pair_base, pair_remainder = divmod(count, len(pair_tuples))
        pair_schedule = pair_tuples * pair_base + rng.sample(
            pair_tuples, pair_remainder
        )
        rng.shuffle(pair_schedule)

        schedules: dict[str, list[str]] = {
            pair_factors[0]: [pair[0] for pair in pair_schedule],
            pair_factors[1]: [pair[1] for pair in pair_schedule],
        }
        for factor_name, required in required_levels.items():
            if factor_name in schedules:
                continue
            levels = sorted(required)
            base, remainder = divmod(count, len(levels))
            schedule = levels * base + rng.sample(levels, remainder)
            rng.shuffle(schedule)
            schedules[factor_name] = schedule

        keys = [
            tuple(schedules[name][index] for name in factor_names)
            for index in range(count)
        ]
        if len(set(keys)) != count or any(
            key not in candidate_keys for key in keys
        ):
            continue
        selected = [dict(candidate_keys[key]) for key in keys]
        if _covers_all_levels(selected, required_levels):
            return selected
    raise ValueError(
        f"Could not select {count} balanced combinations constrained to "
        f"{pair_factors} pairs"
    )


def _orthogonal_combination_sample(
    factor_names: list[str],
    shared_levels: dict[str, list[str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    """Construct a balanced subset for the four-factor, four-level design."""
    level_counts = {len(shared_levels[name]) for name in factor_names}
    if len(factor_names) != 4 or level_counts != {4} or count != 15:
        return _sample_with_coverage(
            _factor_product(factor_names, shared_levels),
            count,
            {name: set(shared_levels[name]) for name in factor_names},
            seed,
            tolerance=1,
        )

    candidates = []
    for first in range(4):
        for second in range(4):
            indices = (
                first,
                second,
                (first + second) % 4,
                (first + 2 * second) % 4,
            )
            candidates.append(
                {
                    factor_name: shared_levels[factor_name][level_index]
                    for factor_name, level_index in zip(factor_names, indices)
                }
            )
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def build_dataset_splits(
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic train and three evaluation combination sets."""
    factors = config["factors"]
    factor_names = [factor["name"] for factor in factors]
    factor_lookup = {factor["name"]: factor for factor in factors}
    level_lookup = {
        factor["name"]: {
            level["name"]: level for level in factor["levels"]
        }
        for factor in factors
    }
    shared_levels = {
        factor["name"]: [
            level["name"]
            for level in factor["levels"]
            if level.get("availability", "shared") == "shared"
        ]
        for factor in factors
    }
    shared_level_sets = {
        factor_name: set(levels)
        for factor_name, levels in shared_levels.items()
    }
    shared_pool = _factor_product(factor_names, shared_levels)
    train_count = config["splits"]["train"]["num_combinations"]
    combination_test_count = config["splits"]["test_combinations"][
        "num_combinations"
    ]
    if train_count + combination_test_count > len(shared_pool):
        raise ValueError("Not enough shared-level combinations for train and test")

    combination_seed = config["combination_seed"]
    pair_holdout = config["splits"]["test_combinations"]["holdout"].get(
        "factor_pairs"
    )
    if pair_holdout is None:
        combination_test_specs = _orthogonal_combination_sample(
            factor_names,
            shared_levels,
            combination_test_count,
            combination_seed,
        )
    else:
        pair_factors = pair_holdout["factors"]
        held_out_pairs = {tuple(pair) for pair in pair_holdout["values"]}
        pair_test_pool = [
            specification
            for specification in shared_pool
            if tuple(specification[name] for name in pair_factors)
            in held_out_pairs
        ]
        combination_test_specs = _sample_with_pair_coverage(
            candidates=pair_test_pool,
            count=combination_test_count,
            required_levels=shared_level_sets,
            pair_factors=pair_factors,
            pairs=pair_holdout["values"],
            seed=combination_seed,
        )
    test_level_tuples = {
        tuple(specification[name] for name in factor_names)
        for specification in combination_test_specs
    }
    remaining_shared_pool = [
        specification
        for specification in shared_pool
        if tuple(specification[name] for name in factor_names)
        not in test_level_tuples
        and (
            pair_holdout is None
            or tuple(specification[name] for name in pair_factors)
            not in held_out_pairs
        )
    ]
    if pair_holdout is None:
        train_specs = _sample_with_coverage(
            remaining_shared_pool,
            train_count,
            shared_level_sets,
            combination_seed + 1_000,
            tolerance=4,
        )
    else:
        allowed_train_pairs = [
            list(pair)
            for pair in itertools.product(
                *(shared_levels[name] for name in pair_factors)
            )
            if pair not in held_out_pairs
        ]
        train_specs = _sample_with_pair_coverage(
            candidates=remaining_shared_pool,
            count=train_count,
            required_levels=shared_level_sets,
            pair_factors=pair_factors,
            pairs=allowed_train_pairs,
            seed=combination_seed + 1_000,
        )

    feature_specs: list[dict[str, str]] = []
    feature_groups = config["splits"]["test_features"]["holdout"]["groups"]
    for group_index, group in enumerate(feature_groups):
        group_levels = {
            factor_name: list(levels)
            for factor_name, levels in shared_levels.items()
        }
        group_levels[group["factor"]] = list(group["levels"])
        group_candidates = _factor_product(factor_names, group_levels)
        if pair_holdout is not None:
            group_candidates = [
                specification
                for specification in group_candidates
                if tuple(specification[name] for name in pair_factors)
                not in held_out_pairs
            ]
        group_required = {
            factor_name: (
                set(group["levels"])
                if factor_name == group["factor"]
                else set(shared_levels[factor_name])
            )
            for factor_name in factor_names
        }
        selected = _sample_with_coverage(
            group_candidates,
            group["num_combinations"],
            group_required,
            combination_seed + 10_000 * (group_index + 1),
        )
        for specification in selected:
            specification = dict(specification)
            specification["_holdout_factor"] = group["factor"]
            feature_specs.append(specification)

    def materialize(
        split_name: str, specifications: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        combinations = []
        for index, specification in enumerate(specifications):
            level_names = {
                name: specification[name] for name in factor_names
            }
            variation_values: dict[str, Any] = {}
            position_regions: dict[str, dict[str, Any]] = {}
            for factor_name in factor_names:
                factor = factor_lookup[factor_name]
                level_name = level_names[factor_name]
                level = level_lookup[factor_name][level_name]
                if factor.get("kind", "variation") == "variation":
                    variation_values[factor_name] = level["value"]
                else:
                    position_regions[factor["variation"]] = {
                        "factor": factor_name,
                        "region": level_name,
                        "low": level["low"],
                        "high": level["high"],
                    }
            combination = {
                "id": f"{split_name}_{index:04d}",
                "levels": level_names,
                "variation_values": variation_values,
            }
            if position_regions:
                combination["position_regions"] = position_regions
            if "_holdout_factor" in specification:
                combination["holdout_factor"] = specification[
                    "_holdout_factor"
                ]
            combinations.append(combination)
        return combinations

    splits = {
        "train": materialize("train", train_specs),
        "test_indistribution": materialize(
            "test_indistribution", train_specs
        ),
        "test_combinations": materialize(
            "test_combinations", combination_test_specs
        ),
        "test_features": materialize("test_features", feature_specs),
    }
    _validate_dataset_splits(config, splits)
    return splits


def _validate_dataset_splits(
    config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    factor_names = [factor["name"] for factor in config["factors"]]
    level_tuples = {
        split_name: {
            tuple(combination["levels"][name] for name in factor_names)
            for combination in combinations
        }
        for split_name, combinations in splits.items()
    }
    if level_tuples["test_indistribution"] != level_tuples["train"]:
        raise ValueError(
            "test_indistribution must contain exactly the training combinations"
        )
    if (
        level_tuples["train"] & level_tuples["test_combinations"]
        or level_tuples["train"] & level_tuples["test_features"]
        or level_tuples["test_combinations"] & level_tuples["test_features"]
    ):
        raise ValueError("Dataset splits contain overlapping combinations")

    test_only_levels = {
        factor["name"]: {
            level["name"]
            for level in factor["levels"]
            if level.get("availability", "shared") == "test_only"
        }
        for factor in config["factors"]
    }
    for split_name in (
        "train",
        "test_indistribution",
        "test_combinations",
    ):
        for combination in splits[split_name]:
            leaked = {
                factor_name: combination["levels"][factor_name]
                for factor_name, levels in test_only_levels.items()
                if combination["levels"][factor_name] in levels
            }
            if leaked:
                raise ValueError(
                    f"Test-only feature levels leaked into {split_name}: {leaked}"
                )

    for combination in splits["test_features"]:
        holdout_factor = combination["holdout_factor"]
        held_out_value = combination["levels"][holdout_factor]
        if held_out_value not in test_only_levels[holdout_factor]:
            raise ValueError(
                f"Invalid feature holdout {holdout_factor}={held_out_value}"
            )
        for factor_name, levels in test_only_levels.items():
            if factor_name == holdout_factor:
                continue
            if combination["levels"][factor_name] in levels:
                raise ValueError(
                    "Feature test combinations must isolate one held-out "
                    f"factor, but found test-only {factor_name}"
                )

    pair_holdout = config["splits"]["test_combinations"]["holdout"].get(
        "factor_pairs"
    )
    if pair_holdout is not None:
        pair_factors = pair_holdout["factors"]
        held_out_pairs = {tuple(pair) for pair in pair_holdout["values"]}

        def spatial_pair(combination: dict[str, Any]) -> tuple[str, str]:
            return tuple(
                combination["levels"][name] for name in pair_factors
            )

        observed_test_pairs = {
            spatial_pair(combination)
            for combination in splits["test_combinations"]
        }
        if observed_test_pairs != held_out_pairs:
            raise ValueError(
                "test_combinations must cover exactly the declared held-out "
                f"factor pairs; got {sorted(observed_test_pairs)}"
            )
        for split_name in (
            "train",
            "test_indistribution",
            "test_features",
        ):
            leaked_pairs = {
                spatial_pair(combination)
                for combination in splits[split_name]
                if spatial_pair(combination) in held_out_pairs
            }
            if leaked_pairs:
                raise ValueError(
                    f"Combination-held-out factor pairs leaked into {split_name}: "
                    f"{sorted(leaked_pairs)}"
                )


def allocate_episodes(
    combinations: list[dict[str, Any]],
    total_episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Allocate episodes almost uniformly, with deterministic remainder order."""
    if total_episodes < len(combinations):
        raise ValueError(
            f"{total_episodes} episodes cannot cover {len(combinations)} "
            "combinations at least once"
        )

    base, remainder = divmod(total_episodes, len(combinations))
    remainder_order = list(range(len(combinations)))
    random.Random(seed).shuffle(remainder_order)
    receives_extra = set(remainder_order[:remainder])

    allocated = []
    for index, combination in enumerate(combinations):
        item = dict(combination)
        item["episodes"] = base + int(index in receives_extra)
        allocated.append(item)
    return allocated


def _ordered_unique(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(values))


def make_split_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if "all" in args.extra_variations:
        raise ValueError(
            "Use explicit --variations names for a controlled benchmark; "
            "'all' would add uncontrolled factors to the split."
        )
    physics_steps = validate_timesteps(
        args.physics_timestep, args.control_timestep
    )
    config = _load_split_config(args.split_config, args.env_type)
    generated_splits = build_dataset_splits(config)
    episode_totals = {
        "train": args.episodes,
        "test_indistribution": len(
            generated_splits["test_indistribution"]
        ),
        "test_combinations": (
            args.combination_test_episodes or args.test_episodes
        ),
        "test_features": args.feature_test_episodes or args.test_episodes,
    }
    split_combinations = {
        split_name: allocate_episodes(
            combinations,
            episode_totals[split_name],
            args.seed + SPLIT_SEED_OFFSETS[split_name],
        )
        for split_name, combinations in generated_splits.items()
    }

    controlled_factors = [factor["name"] for factor in config["factors"]]
    controlled_variations = [
        factor.get("variation", factor["name"])
        for factor in config["factors"]
    ]
    reset_variations = _ordered_unique(
        list(DEFAULT_TASK_VARIATIONS)
        + controlled_variations
        + args.extra_variations
    )
    split_definitions = copy.deepcopy(config["splits"])
    split_definitions["test_indistribution"] = {
        "source": "train",
        "num_combinations": len(
            split_combinations["test_indistribution"]
        ),
        "episodes_per_combination": 1,
    }
    held_out_test_names = ("test_combinations", "test_features")
    all_test_names = (
        "test_indistribution",
        "test_combinations",
        "test_features",
    )
    manifest = {
        "status": "planned",
        "split_config": str(args.split_config.resolve()),
        "split_name": config.get("name", args.split_config.stem),
        "combination_seed": config["combination_seed"],
        "split_definitions": split_definitions,
        # Keep this legacy field scoped to the two held-out test sets.
        "total_test_combinations": sum(
            len(split_combinations[name])
            for name in held_out_test_names
        ),
        "total_evaluation_combinations": sum(
            len(split_combinations[name])
            for name in all_test_names
        ),
        "controlled_factors": controlled_factors,
        "controlled_variations": controlled_variations,
        "position_sampling": config.get("position_sampling", {}),
        "reset_variations": reset_variations,
        "deterministic_variation_resets": True,
        "timing": {
            "physics_timestep_seconds": args.physics_timestep,
            "control_timestep_seconds": args.control_timestep,
            "control_hz": 1.0 / args.control_timestep,
            "physics_steps_per_control": physics_steps,
        },
        "policy": {
            "type": args.policy_type,
            "action_noise": args.action_noise,
            "p_random_action": args.p_random_action,
            "noise_smoothing": args.noise_smoothing,
            "min_norm": args.min_norm,
        },
        "splits": {
            name: {
                "episodes": sum(item["episodes"] for item in combinations),
                "num_combinations": len(combinations),
                "combinations": combinations,
            }
            for name, combinations in split_combinations.items()
        },
    }
    return manifest, split_combinations


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value.resolve())
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _check_output_targets(args: argparse.Namespace) -> None:
    existing = []
    for split_name in SPLIT_SEED_OFFSETS:
        split_dir = args.output_dir / split_name
        metadata = split_dir / f"{args.task_name}.pt"
        strips = list(split_dir.glob(f"{args.task_name}-*.png"))
        if metadata.exists() or strips:
            existing.append(str(split_dir))
    if existing and not args.overwrite:
        raise FileExistsError(
            "Dataset output already exists in "
            + ", ".join(existing)
            + "; pass --overwrite to replace it"
        )


class _BorrowedWriter:
    """Let World.collect use an already-open writer without closing it."""

    def __init__(self, writer: Any) -> None:
        self.writer = writer

    def __enter__(self) -> Any:
        return self.writer

    def __exit__(self, *exc: object) -> None:
        return None


def _sample_position_region_values(
    *,
    position_regions: dict[str, dict[str, Any]],
    seed: int | None,
    num_cubes: int,
    minimum_start_goal_distance: float,
    max_attempts: int,
) -> dict[str, list[list[float]]]:
    """Sample per-episode coordinates inside categorical spatial regions."""
    required = {"cube.start_position", "cube.goal_position"}
    if set(position_regions) != required:
        raise ValueError(
            "Position-region combinations must define exactly "
            f"{sorted(required)}, got {sorted(position_regions)}"
        )

    rng_seed = None if seed is None else seed ^ 0x5A17_10C
    rng = random.Random(rng_seed)

    def sample_xy(specification: dict[str, Any]) -> list[float]:
        return [
            rng.uniform(lower, upper)
            for lower, upper in zip(
                specification["low"], specification["high"]
            )
        ]

    starts: list[list[float]] = []
    goals: list[list[float]] = []
    start_region = position_regions["cube.start_position"]
    goal_region = position_regions["cube.goal_position"]
    for _ in range(num_cubes):
        for _attempt in range(max_attempts):
            start = sample_xy(start_region)
            goal = sample_xy(goal_region)
            distance = math.hypot(start[0] - goal[0], start[1] - goal[1])
            if distance >= minimum_start_goal_distance:
                starts.append(start)
                goals.append(goal)
                break
        else:
            raise ValueError(
                "Could not sample start and goal positions at least "
                f"{minimum_start_goal_distance} m apart after {max_attempts} "
                f"attempts (start={start_region['region']!r}, "
                f"goal={goal_region['region']!r})"
            )

    return {
        "cube.start_position": starts,
        "cube.goal_position": goals,
    }


def _make_seeded_variation_wrapper(gym: Any) -> type:
    """Materialize reset variations from the episode seed.

    This also works around Stable-WorldModel CubeEnv versions that do not
    forward ``seed`` to their variation space.
    """

    class SeededVariationWrapper(gym.Wrapper):
        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            options = copy.deepcopy(options or {})
            variation_space = self.env.unwrapped.variation_space
            position_regions = options.pop("position_regions", None)
            position_sampling = options.pop("position_sampling", {})
            variation_keys = options.get(
                "variation", list(DEFAULT_TASK_VARIATIONS)
            )

            sampled_values = {}
            if seed is not None:
                variation_space.seed(seed)
                variation_space.reset()
                variation_space.update(variation_keys)
                for key in variation_keys:
                    value_space = variation_space
                    for part in key.split("."):
                        value_space = value_space[part]
                    sampled_values[key] = copy.deepcopy(value_space.value)

            # Exact compositional levels take precedence over sampled variations.
            sampled_values.update(options.get("variation_values", {}))
            if position_regions is not None:
                num_cubes = variation_space["cube"][
                    "start_position"
                ].shape[0]
                sampled_values.update(
                    _sample_position_region_values(
                        position_regions=position_regions,
                        seed=seed,
                        num_cubes=num_cubes,
                        minimum_start_goal_distance=position_sampling.get(
                            "minimum_start_goal_distance", 0.0
                        ),
                        max_attempts=position_sampling.get("max_attempts", 100),
                    )
                )

            if sampled_values:
                options["variation_values"] = sampled_values

            return self.env.reset(seed=seed, options=options)

    return SeededVariationWrapper


def _collect_split(
    *,
    world: Any,
    get_format: Any,
    args: argparse.Namespace,
    split_name: str,
    combinations: list[dict[str, Any]],
    reset_variations: list[str],
    position_sampling: dict[str, Any],
) -> None:
    split_dir = args.output_dir / split_name
    episode_cursor = 0
    mode = "overwrite" if args.overwrite else "error"
    writer_cm = get_format("dreamer_folder").open_writer(
        split_dir,
        task_name=args.task_name,
        mode=mode,
    )
    with writer_cm as writer:
        borrowed_writer = _BorrowedWriter(writer)
        for combination in combinations:
            options = {
                "variation": reset_variations,
                "variation_values": combination["variation_values"],
            }
            if "position_regions" in combination:
                options["position_regions"] = combination[
                    "position_regions"
                ]
                options["position_sampling"] = position_sampling
            collection_seed = (
                args.seed
                + SPLIT_SEED_OFFSETS[split_name]
                + episode_cursor
            )
            print(
                f"[{split_name}] {combination['id']}: "
                f"{combination['episodes']} episodes, "
                f"levels={combination['levels']}"
            )
            world.collect(
                writer=borrowed_writer,
                episodes=combination["episodes"],
                seed=collection_seed,
                options=options,
                progress=not args.no_progress,
            )
            episode_cursor += combination["episodes"]


def generate_data() -> None:
    args = parse_args()
    manifest, split_combinations = make_split_plan(args)


    _check_output_targets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "dataset_config.json", _serializable_args(args))
    _write_json(args.output_dir / "split_manifest.json", manifest)

    import stable_worldmodel as swm
    import gymnasium as gym
    from stable_worldmodel.data.format import get_format
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    # Import for its @register_format side effect.
    import dataset.dreamer_folder_dataset  # noqa: F401

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        env_type=args.env_type,
        image_shape=(args.image_shape, args.image_shape),
        max_episode_steps=args.max_episode_steps,
        pre_wrappers=[_make_seeded_variation_wrapper(gym)],
        physics_timestep=args.physics_timestep,
        control_timestep=args.control_timestep,
        mode="data_collection",
    )
    policy = ExpertPolicy(
        policy_type=args.policy_type,
        action_noise=args.action_noise,
        p_random_action=args.p_random_action,
        noise_smoothing=args.noise_smoothing,
        min_norm=args.min_norm,
        seed=args.seed,
    )
    world.set_policy(policy)

    try:
        reset_variations = manifest["reset_variations"]
        for split_name in (
            "train",
            "test_combinations",
            "test_features",
            "test_indistribution",
        ):
            _collect_split(
                world=world,
                get_format=get_format,
                args=args,
                split_name=split_name,
                combinations=split_combinations[split_name],
                reset_variations=reset_variations,
                position_sampling=manifest["position_sampling"],
            )
    except Exception:
        manifest["status"] = "incomplete"
        _write_json(args.output_dir / "split_manifest.json", manifest)
        raise
    finally:
        world.close()

    manifest["status"] = "complete"
    _write_json(args.output_dir / "split_manifest.json", manifest)
    print(f"Finished dataset at {args.output_dir}")


if __name__ == "__main__":
    generate_data()