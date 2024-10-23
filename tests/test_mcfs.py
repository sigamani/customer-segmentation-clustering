import pandas as pd
try:
    from app.feature_selection.feature_selection import get_multi_cluster
except ModuleNotFoundError:
    from ..app.feature_selection.feature_selection import get_multi_cluster

""" These file names are unnecessarily long"""

df = pd.read_parquet('s3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/'
                     'qudo_financialservicesfinal_uk_q1_2023_staging/'
                     'qudo_financialservicesfinal_uk_q1_2023_staging_responses/'
                     'qudo_financialservicesfinal_uk_q1_2023_staging_responses.parquet')


def test_results_match():
    test_obj = get_multi_cluster(df, n_features=50, n_neighbours=15, t=0.05, n_clusters=7, return_ranking=True)
    assert isinstance(test_obj, list)
