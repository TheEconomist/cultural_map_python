from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VARS = [
    "A008",
    "A165",
    "E018",
    "E025",
    "F063",
    "F118",
    "F120",
    "G006",
    "Y002",
    "Y003",
]
PAIRWISE_MISSING_VARS = ANALYSIS_VARS[:-1]
Y003_MISSING_VAR = "Y003"
DEFAULT_INPUT_CANDIDATES = [
    Path(__file__).resolve().parent / "../data/wvs_evs_time_series.csv",
    Path(__file__).resolve().parent / "data/wvs_evs_time_series.csv",
]
OUTPUT_PATH: Path | None = None
WEIGHT_VAR = "S017"


def resolve_input_path() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path.resolve()) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Input dataset not found. Checked:\n{searched}")


def apply_spss_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[WEIGHT_VAR] = pd.to_numeric(data[WEIGHT_VAR], errors="coerce")
    data[ANALYSIS_VARS] = data[ANALYSIS_VARS].apply(pd.to_numeric, errors="coerce")
    for column in PAIRWISE_MISSING_VARS:
        data[column] = data[column].mask(data[column].between(-9, -1))
    data[Y003_MISSING_VAR] = data[Y003_MISSING_VAR].mask(data[Y003_MISSING_VAR] == -5)
    return data


def pca_loadings(corr: np.ndarray, n_factors: int = 2) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_factors]
    eigenvectors = eigenvectors[:, order][:, :n_factors]
    loadings = eigenvectors * np.sqrt(eigenvalues)
    return eigenvalues, loadings


def varimax_kaiser(
    loadings: np.ndarray,
    gamma: float = 1.0,
    q: int = 25,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    p, k = loadings.shape
    communalities = np.sqrt((loadings**2).sum(axis=1))
    communalities[communalities == 0] = 1.0

    normalized = loadings / communalities[:, None]
    rotation = np.eye(k)
    prev_sum = 0.0

    for _ in range(q):
        rotated = normalized @ rotation
        basis = normalized.T @ (
            rotated**3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
        )
        u, singular_values, vh = np.linalg.svd(basis)
        rotation = u @ vh
        current_sum = singular_values.sum()
        if prev_sum and current_sum - prev_sum < tol:
            break
        prev_sum = current_sum

    return (normalized @ rotation) * communalities[:, None], rotation


def orient_rotated_factors(rotated_loadings: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    y002_index = ANALYSIS_VARS.index("Y002")
    surv_idx = int(np.argmax(np.abs(rotated_loadings[y002_index, :])))
    trad_idx = 1 - surv_idx

    oriented = rotated_loadings.copy()

    if oriented[ANALYSIS_VARS.index("F118"), surv_idx] < 0:
        oriented[:, surv_idx] *= -1
    if oriented[ANALYSIS_VARS.index("A008"), trad_idx] < 0:
        oriented[:, trad_idx] *= -1

    return oriented, {"survself": surv_idx, "tradrat5": trad_idx}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(weights * values) / np.sum(weights)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    variance = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
    return float(np.sqrt(variance))


def weighted_pairwise_correlation_matrix(data: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    corr = pd.DataFrame(np.eye(len(ANALYSIS_VARS)), index=ANALYSIS_VARS, columns=ANALYSIS_VARS, dtype=float)

    for i, left in enumerate(ANALYSIS_VARS):
        for j in range(i + 1, len(ANALYSIS_VARS)):
            right = ANALYSIS_VARS[j]
            pair = data[[left, right]].copy()
            pair[WEIGHT_VAR] = weights
            pair = pair.dropna()
            pair = pair[pair[WEIGHT_VAR] > 0]

            if pair.empty:
                value = np.nan
            else:
                x = pair[left].to_numpy(dtype=float)
                y = pair[right].to_numpy(dtype=float)
                w = pair[WEIGHT_VAR].to_numpy(dtype=float)
                mean_x = weighted_mean(x, w)
                mean_y = weighted_mean(y, w)
                cov_xy = np.sum(w * (x - mean_x) * (y - mean_y)) / np.sum(w)
                var_x = np.sum(w * (x - mean_x) ** 2) / np.sum(w)
                var_y = np.sum(w * (y - mean_y) ** 2) / np.sum(w)
                denom = np.sqrt(var_x * var_y)
                value = np.nan if denom == 0 else cov_xy / denom

            corr.loc[left, right] = value
            corr.loc[right, left] = value

    return corr


def regression_scores(
    clean_data: pd.DataFrame,
    case_weights: pd.Series,
    corr: np.ndarray,
    rotated_loadings: np.ndarray,
) -> pd.DataFrame:
    means = {}
    stds = {}
    for column in ANALYSIS_VARS:
        valid = clean_data[column].notna() & case_weights.notna() & (case_weights > 0)
        values = clean_data.loc[valid, column].to_numpy(dtype=float)
        weights = case_weights.loc[valid].to_numpy(dtype=float)
        means[column] = weighted_mean(values, weights)
        stds[column] = weighted_std(values, weights)

    means = pd.Series(means)
    stds = pd.Series(stds)
    standardized = (clean_data[ANALYSIS_VARS] - means) / stds
    complete_case_mask = standardized.notna().all(axis=1)

    corr_inv = np.linalg.inv(corr)
    weights = corr_inv @ rotated_loadings @ np.linalg.inv(rotated_loadings.T @ corr_inv @ rotated_loadings)

    score_array = np.full((len(clean_data), 2), np.nan)
    score_array[complete_case_mask.to_numpy(), :] = standardized.loc[complete_case_mask].to_numpy() @ weights

    scores = pd.DataFrame(score_array, columns=["fac1_1", "fac2_1"], index=clean_data.index)
    return scores


def blank_small_loadings(loadings: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    formatted = loadings.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.notna(value) and abs(value) < threshold else f"{value:.6f}"
        )
    return formatted


def build_means_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s025, group in df.groupby("S025", dropna=False):
        weights = pd.to_numeric(group[WEIGHT_VAR], errors="coerce")
        entry: dict[str, float | int] = {"S025": s025}
        for source, label in [("TradAgg", "Mean TradAgg"), ("SurvSAgg", "Mean SurvSAgg")]:
            valid = group[source].notna() & weights.notna() & (weights > 0)
            if valid.any():
                values = group.loc[valid, source].to_numpy(dtype=float)
                current_weights = weights.loc[valid].to_numpy(dtype=float)
                entry[label] = weighted_mean(values, current_weights)
            else:
                entry[label] = np.nan
        rows.append(entry)

    return pd.DataFrame(rows).set_index("S025")


def main() -> None:
    input_path = resolve_input_path()

    raw = pd.read_csv(input_path, low_memory=False, na_values=["", " "])
    clean = apply_spss_missing_values(raw)
    case_weights = clean[WEIGHT_VAR]

    corr_df = weighted_pairwise_correlation_matrix(clean[ANALYSIS_VARS], case_weights)
    eigenvalues, initial_loadings = pca_loadings(corr_df.to_numpy(), n_factors=2)
    rotated_loadings, _ = varimax_kaiser(initial_loadings, q=25)
    rotated_loadings, factor_map = orient_rotated_factors(rotated_loadings)

    scores = regression_scores(clean, case_weights, corr_df.to_numpy(), rotated_loadings)
    renamed_scores = pd.DataFrame(index=scores.index)
    renamed_scores["survself"] = scores.iloc[:, factor_map["survself"]]
    renamed_scores["tradrat5"] = scores.iloc[:, factor_map["tradrat5"]]

    result = raw.copy()
    result["survself"] = renamed_scores["survself"]
    result["tradrat5"] = renamed_scores["tradrat5"]
    result["TradAgg"] = 1.61 * result["tradrat5"] - 0.1
    result["SurvSAgg"] = 1.81 * result["survself"] + 0.038

    initial_loadings_df = pd.DataFrame(
        initial_loadings,
        index=ANALYSIS_VARS,
        columns=["Component 1", "Component 2"],
    )
    rotated_loadings_df = pd.DataFrame(
        rotated_loadings,
        index=ANALYSIS_VARS,
        columns=["Factor 1", "Factor 2"],
    )
    means_table = build_means_table(result)

    print(f"Input dataset: {input_path}")
    print()
    print("Eigenvalues")
    print(pd.Series(eigenvalues, index=["Component 1", "Component 2"]).to_string(float_format=lambda x: f"{x:.6f}"))
    print()
    print("Initial component loadings")
    print(initial_loadings_df.to_string(float_format=lambda x: f"{x:.6f}"))
    print()
    print("Rotated loadings (Varimax with Kaiser normalization, blank if |loading| < .3)")
    print(blank_small_loadings(rotated_loadings_df).to_string())
    print()
    print("Means table by S025")
    print(means_table.to_string(float_format=lambda x: f"{x:.6f}"))

    if OUTPUT_PATH is not None:
        output_path = OUTPUT_PATH.expanduser().resolve()
        result.to_csv(output_path, index=False)
        print()
        print(f"Saved rescored dataset to: {output_path}")


if __name__ == "__main__":
    main()
