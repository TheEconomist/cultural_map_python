from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from cultural_map import build_cultural_map


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wvs_evs_time_series.csv"


class BuildCulturalMapTest(unittest.TestCase):
    def test_computed_aggregates_match_existing_dataset_columns(self) -> None:
        original = pd.read_csv(DATA_PATH, low_memory=False, na_values=["", " "])
        expected_tradagg = original["TradAgg"].copy()
        expected_survsagg = original["SurvSAgg"].copy()

        computed_df, _ = build_cultural_map(original.copy())

        assert_series_equal(
            computed_df["TradAgg"],
            expected_tradagg,
            check_names=False,
            check_exact=False,
            atol=1e-4,
            rtol=0.0,
        )
        assert_series_equal(
            computed_df["SurvSAgg"],
            expected_survsagg,
            check_names=False,
            check_exact=False,
            atol=1e-4,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
