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
WEIGHT_VAR = "S017"


def apply_spss_missing_values(df: pd.DataFrame):
    for column in ANALYSIS_VARS[:-1]:
        df[column] = df[column].mask(df[column].between(-9, -1))
    df["Y003"] = df["Y003"].mask(df["Y003"] == -5)
    return df


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
    communalities = np.sqrt((loadings ** 2).sum(axis=1))
    communalities[communalities == 0] = 1.0

    normalized = loadings / communalities[:, None]
    rotation = np.eye(k)
    prev_sum = 0.0

    for _ in range(q):
        rotated = normalized @ rotation
        basis = normalized.T @ (
                rotated ** 3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
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
) -> np.ndarray:
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

    return score_array


def build_means_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df.dropna(subset=["TradAgg", WEIGHT_VAR])
        .groupby("S025")
        .apply(lambda x: np.average(x["TradAgg"], weights=x[WEIGHT_VAR]))
        .rename("Mean TradAgg"),
        df.dropna(subset=["SurvSAgg", WEIGHT_VAR])
        .groupby("S025")
        .apply(lambda x: np.average(x["SurvSAgg"], weights=x[WEIGHT_VAR]))
        .rename("Mean SurvSAgg"),
    ], axis=1)


def build_cultural_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    A faithful reproduction of the SPSS cultural map syntax found on the WVS website.
    """

    """
    MISSING VALUES a008 a165 e018 e025 f063 f118 f120 g006 y002 (-9 to -1).
    MISSING  VALUES y003 (-5).
    """
    apply_spss_missing_values(df)

    """
    /MISSING  PAIRWISE
    """
    case_weights = df[WEIGHT_VAR]
    corr_df = weighted_pairwise_correlation_matrix(df[ANALYSIS_VARS], case_weights)

    """
    /CRITERIA FACTORS(2) ITERATE(25)
    /EXTRACTION PC
    /CRITERIA ITERATE(25)
    /ROTATION VARIMAX
    /SAVE REG(ALL)
    /METHOD=CORRELATION.
    RENAME VARIABLES (fac2_1=tradrat5) (fac1_1=survself).
    """
    eigenvalues, initial_loadings = pca_loadings(corr_df.to_numpy(), n_factors=2)
    rotated_loadings, _ = varimax_kaiser(initial_loadings, q=25)
    rotated_loadings, factor_map = orient_rotated_factors(rotated_loadings)
    score_array = regression_scores(df, case_weights, corr_df.to_numpy(), rotated_loadings)
    scores = pd.DataFrame(score_array, columns=["survself", "tradrat5"], index=df.index)

    """
    COMPUTE TradAgg = 1.61 * TradRat5  - .1 .
    COMPUTE SurvSAgg = 1.81 * SurvSelf  + .038 .
    """
    df["TradAgg"] = 1.61 * scores["tradrat5"] - 0.1
    df["SurvSAgg"] = 1.81 * scores["survself"] + 0.038

    """
    MEANS TABLES=TradAgg SurvSAgg BY S025 /CELLS MEAN.
    """
    means_table = build_means_table(df)
    return means_table


if __name__ == "__main__":
    df = pd.read_csv("../data/wvs_evs_time_series.csv", low_memory=False, na_values=["", " "])
    cultural_map = build_cultural_map(df)
    cultural_map.to_csv("../output/cultural_map.csv")
