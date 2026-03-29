import numpy as np
import krippendorff

# ==== USER INPUT SECTION =====================================
# Edit these dictionaries with your actual scores (1-5) per metric.
# Each metric maps to a dict: unit_name -> [rater1, rater2, rater3]
# Use None for missing ratings.

# The score are in the order of Gemini -> Claude -> Grok -> Deepseek -> Meta LLama 4 Thinking

EXTENSIBILITY_SCORES = {
    'base_equi': [1, 2, 2, 1, 2],
    'multi_equi': [3, 4, 4, 4, 4],
    'base_sen': [2, 3, 3, 3, 3],
    'multi_sen': [3, 4, 4, 4, 4],
}

REDUNDANCY_SCORES = {
    'base_equi': [1, 1, 1, 1, 1],
    'multi_equi': [2, 3, 3, 2, 2],
    'base_sen': [2, 2, 2, 2, 2],
    'multi_sen': [1, 2, 3, 2, 1],
}

ODP_USAGE_SCORES = {
    'base_equi': [1, 2, 1, 1, 2],
    'multi_equi': [4, 4, 4, 3, 4],
    'base_sen': [3, 2, 2, 2, 2],
    'multi_sen': [3, 4, 4, 2, 4],
}

# =============================================================

METRICS = {
    "Extensibility": EXTENSIBILITY_SCORES,
    "Redundancy": REDUNDANCY_SCORES,
    "ODP_Usage": ODP_USAGE_SCORES,
}


def dict_to_matrix(metric_dict):
    """Convert dict of unit -> [r1, r2, r3] to 2D array (raters x units)."""
    units = list(metric_dict.keys())
    if not units:
        return None, []
    data = np.array([metric_dict[u] for u in units], dtype=float).T
    return data, units


def alpha_for_metric(name, metric_dict, level_of_measurement="ordinal"):
    data, units = dict_to_matrix(metric_dict)
    if data is None:
        return None, units
    a = krippendorff.alpha(
        reliability_data=data,
        level_of_measurement=level_of_measurement
    )
    return a, units


def main():
    print("=== Krippendorff's Alpha (ordinal, 1-5) ===\n")

    # Per-metric alphas
    per_metric_results = {}
    for metric_name, scores in METRICS.items():
        alpha, units = alpha_for_metric(metric_name, scores)
        per_metric_results[metric_name] = (alpha, units)

    # Print per-metric table
    print("Per-metric reliability (Krippendorff's alpha):")
    print("{:<15} {:>8}   Units".format("Metric", "Alpha"))
    print("-" * 40)
    for metric_name, (alpha, units) in per_metric_results.items():
        alpha_str = "NA" if alpha is None else f"{alpha:0.3f}"
        units_str = ", ".join(units) if units else "(none)"
        print(f"{metric_name:<15} {alpha_str:>8}   {units_str}")

    # Overall alpha across all metrics
    # Stack all units from all metrics into a single matrix
    all_units = []
    all_data_cols = []
    for metric_name, scores in METRICS.items():
        for unit, ratings in scores.items():
            all_units.append(f"{metric_name}:{unit}")
            all_data_cols.append(ratings)

    if all_data_cols:
        all_data = np.array(all_data_cols, dtype=float).T
        overall_alpha = krippendorff.alpha(
            reliability_data=all_data,
            level_of_measurement="ordinal"
        )
        print("\nOverall reliability across all metrics and ontologies:")
        print("{:<25} {:>8}".format("Overall Alpha", f"{overall_alpha:0.3f}"))
    else:
        print("\nNo data provided; fill in the *_SCORES dictionaries above.")


if __name__ == "__main__":
    main()
