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


def resolve_input_path() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path.resolve()) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Input dataset not found. Checked:\n{searched}")


def apply_spss_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
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


def regression_scores(
    clean_data: pd.DataFrame,
    corr: np.ndarray,
    rotated_loadings: np.ndarray,
) -> pd.DataFrame:
    means = clean_data[ANALYSIS_VARS].mean()
    stds = clean_data[ANALYSIS_VARS].std(ddof=0)
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
    means_table = (
        df.groupby("S025", dropna=False)[["TradAgg", "SurvSAgg"]]
        .mean()
        .rename(columns={"TradAgg": "Mean TradAgg", "SurvSAgg": "Mean SurvSAgg"})
    )
    return means_table


def main() -> None:
    input_path = resolve_input_path()

    raw = pd.read_csv(input_path, low_memory=False, na_values=["", " "])
    clean = apply_spss_missing_values(raw)

    corr_df = clean[ANALYSIS_VARS].corr()
    eigenvalues, initial_loadings = pca_loadings(corr_df.to_numpy(), n_factors=2)
    rotated_loadings, _ = varimax_kaiser(initial_loadings, q=25)
    rotated_loadings, factor_map = orient_rotated_factors(rotated_loadings)

    scores = regression_scores(clean, corr_df.to_numpy(), rotated_loadings)
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
