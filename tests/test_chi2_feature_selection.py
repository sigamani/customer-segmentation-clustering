import numpy as np
import pytest
import pandas as pd
from ..app.feature_selection.feature_selection import get_chi_squared_ranking, encode_cols

fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_preprocessed/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses.parquet'
rdf = pd.read_parquet(fin_ser)
sbeh_cols = [x for x in rdf.columns if any(specific_col in x for specific_col in ['sbeh', 'aida'])]
rdf = rdf[sbeh_cols]
enc = rdf.apply(encode_cols)
chi2_data_kprototypes = get_chi_squared_ranking(rdf)
chi2_data_kmeans = get_chi_squared_ranking(enc, 'kmeans')


def test_feature_ranking_kmeans():
    for idx, value in chi2_data_kmeans.items():
        assert value >= 0


def test_feature_ranking_kporototypes():
    for idx, value in chi2_data_kprototypes.items():
        assert value >= 0

def test_false_data():
    empty_df = pd.DataFrame()
    wrong_string = 'Mohsen Mohsen'
    test_1 = get_chi_squared_ranking(empty_df, wrong_string)
    test_2 = get_chi_squared_ranking(empty_df)
    test_3 = get_chi_squared_ranking(empty_df, 'kmeans')
    assert np.isnan(test_1)
    assert np.isnan(test_2)
    assert np.isnan(test_3)

