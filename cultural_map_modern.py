from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator
from sklearn.preprocessing import StandardScaler
from weightedpca import WeightedPCA

ANALYSIS_VARS = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
WEIGHT_VAR = "S017"
GROUP_VAR = "S025"
INPUT_PATH = Path("data/wvs_evs_time_series.csv")
OUTPUT_PATH = Path("output/cultural_map_modern.csv")


def build_modern_cultural_map() -> None:
    """
    Codex's take on a new, modern implementation of the cultural map.
    Produces different results.
    """
    df = pd.read_csv(INPUT_PATH, low_memory=False, na_values=["", " "])
    df[WEIGHT_VAR] = pd.to_numeric(df[WEIGHT_VAR], errors="coerce")
    df[ANALYSIS_VARS] = df[ANALYSIS_VARS].apply(pd.to_numeric, errors="coerce")

    for column in ANALYSIS_VARS[:-1]:
        df[column] = df[column].mask(df[column].between(-9, -1))
    df["Y003"] = df["Y003"].mask(df["Y003"] == -5)

    analysis = df.loc[(df[WEIGHT_VAR] > 0) & df[ANALYSIS_VARS].notna().all(axis=1), [WEIGHT_VAR] + ANALYSIS_VARS].copy()

    scaler = StandardScaler()
    scaler.fit(analysis[ANALYSIS_VARS], sample_weight=analysis[WEIGHT_VAR])
    scaled = scaler.transform(analysis[ANALYSIS_VARS])

    pca = WeightedPCA(n_components=2)
    pca.fit(scaled, sample_weight=analysis[WEIGHT_VAR])
    loadings_array = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotated_loadings = Rotator(method="varimax").fit_transform(loadings_array)
    rotation = np.linalg.lstsq(loadings_array, rotated_loadings, rcond=None)[0]
    scores = pca.transform(scaled) @ rotation
    loadings = pd.DataFrame(
        rotated_loadings,
        index=ANALYSIS_VARS,
        columns=["component_1", "component_2"],
    )

    surv_col = loadings.loc["Y002"].abs().idxmax()
    trad_col = "component_1" if surv_col == "component_2" else "component_2"
    surv_idx = 0 if surv_col == "component_1" else 1
    trad_idx = 0 if trad_col == "component_1" else 1

    if loadings.loc["F118", surv_col] < 0:
        loadings[surv_col] *= -1
        scores[:, surv_idx] *= -1
    if loadings.loc["A008", trad_col] < 0:
        loadings[trad_col] *= -1
        scores[:, trad_idx] *= -1

    result = df[[WEIGHT_VAR, GROUP_VAR]].copy()
    result["survself"] = np.nan
    result["tradrat5"] = np.nan
    result.loc[analysis.index, "survself"] = scores[:, surv_idx]
    result.loc[analysis.index, "tradrat5"] = scores[:, trad_idx]
    result["TradAgg"] = 1.61 * result["tradrat5"] - 0.1
    result["SurvSAgg"] = 1.81 * result["survself"] + 0.038

    means = pd.concat([
        result.dropna(subset=["TradAgg", WEIGHT_VAR])[[GROUP_VAR, "TradAgg", WEIGHT_VAR]]
        .groupby(GROUP_VAR)
        .apply(lambda x: np.average(x["TradAgg"], weights=x[WEIGHT_VAR]), include_groups=False)
        .rename("Mean TradAgg"),
        result.dropna(subset=["SurvSAgg", WEIGHT_VAR])[[GROUP_VAR, "SurvSAgg", WEIGHT_VAR]]
        .groupby(GROUP_VAR)
        .apply(lambda x: np.average(x["SurvSAgg"], weights=x[WEIGHT_VAR]), include_groups=False)
        .rename("Mean SurvSAgg"),
    ], axis=1).dropna(how="any")
    means.index.name = "S025"
    means = means.reset_index()
    means.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    build_modern_cultural_map()
