import pytest
import pandas as pd
from ..app.feature_selection.feature_selection import *


def remove_na_strings_and_floats(col):
    if col.dtypes == 'string':
        col = col.fillna('None')
    else:
        col = col.fillna(-999)
    return col


fin_ser_new = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
fin_ser_preprocessed_new = 's3://qudo-datascience/data-store/surveys/staging/responses_preprocessed/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
preprocessed_df = pd.read_parquet(fin_ser_preprocessed_new)
preprocessed_df = preprocessed_df.apply(remove_na_strings_and_floats)


def test_feature_ranking():
    cols = rank_features(preprocessed_df, 40, 5, 1, 4)
    assert len(cols) <= 40


def test_feature_wrong_data():
    dummy_data = list()
    with pytest.raises(ValueError, match='n_features is more than number of columns'):
        rank_features(preprocessed_df, 9000, 5, 1, 4)
    with pytest.raises(TypeError, match='n_features must be positive Integer'):
        rank_features(preprocessed_df, 90.5, 5, 1, 4)
    with pytest.raises(ValueError, match='n_features must be positive Integer'):
        rank_features(preprocessed_df, -10, 5, 1, 4)
    with pytest.raises(TypeError, match='input should be pandas DataFrame'):
        rank_features(dummy_data, 40, 5, 1, 4)
    with pytest.raises(TypeError, match='n_neighbours must be a positive Integer.'):
        rank_features(preprocessed_df, 40, 90.5, 1, 4)
    with pytest.raises(ValueError, match='n_neighbours must be a positive Integer.'):
        rank_features(preprocessed_df, 40, -10, 1, 4)
        rank_features(preprocessed_df, 40, 90.5, 1, 4)
    with pytest.raises(TypeError, match='n_clusters must be positive Integer'):
        rank_features(preprocessed_df, 40, 5, 1, 10.5)
    with pytest.raises(ValueError, match='n_clusters must be positive Integer'):
        rank_features(preprocessed_df, 40, 5, 1, -10)
