"""
Either run `pytest` or `pytest modules/chisquared_tester/` from your terminal.

P.S. Runs really slow - 1:11:05

If you only want to test specific method please indicate their mark e.g. `pytest -v -m frequencies`

IMPORTANT: To do a soft test assessment of the entire class running `pytest -v -m post_hoc` suffices. When failed one
may run whole test suite. """
import json
import pickle

import numpy as np
import pandas as pd

from ..app.inference.chisquared_tester import ChiSquaredTester

part_dem = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/aa_participatorydemocracy_morocco_q4_2022/aa_participatorydemocracy_morocco_q4_2022_responses/aa_participatorydemocracy_morocco_q4_2022_responses.parquet')
with open('local/data/cluster_cols/part_dem.json', 'r') as f:
    part_dem_cols = json.load(f)

cols = part_dem_cols
cols = [x.lower() for x in cols]
cols_for_clustering = [col for col in part_dem.columns for x in cols if x in col]
df = part_dem.loc[:, ~part_dem.columns.duplicated()].copy()

with open('tests/data_for_tests/ChiSquareTester_test_data.pickle', 'rb') as f:
    k_pca = pickle.load(f)

n_clusters = len(np.unique(k_pca['labels']))
tester = k_pca['inference_data']

clustered_data = df
clustered_data['cluster'] = k_pca['labels']

chi_squared_tester = ChiSquaredTester(clustered_data, 'cluster', 0.95, None, None)
crosstable = chi_squared_tester.crosstab(var='sbeh_ma_govimage_sd_17082022_democracy')
expected = chi_squared_tester.expected_crosstab(crosstab_input=crosstable.copy(deep=True))


def test_crosstab1():
    assert isinstance(crosstable, pd.DataFrame)
    assert n_clusters == crosstable.shape[1]
    assert isinstance(crosstable.iloc[0, 1], np.int64)
    assert crosstable.iloc[0, 0] == 168


def test_crosstab_percent1():
    crosstable_perc = chi_squared_tester.crosstab_percent(crosstab_input=crosstable.copy(deep=True))
    assert isinstance(crosstable_perc, pd.DataFrame)
    assert n_clusters == crosstable_perc.shape[1]
    assert isinstance(crosstable_perc.iloc[0, 1], float)
    assert crosstable_perc.iloc[0, 1] == 28.3


def test_expected_crosstab1():
    assert isinstance(expected, pd.DataFrame)
    assert n_clusters == expected.shape[1]
    assert isinstance(expected.iloc[0, 1], float)
    assert round(expected.iloc[0, 0], 2) == 174.1


def test_direction1():
    direction = chi_squared_tester.direction(crosstab_input=crosstable.copy(deep=True), expected_crosstab=expected)
    assert isinstance(direction, pd.DataFrame)
    assert n_clusters == direction.shape[1]
    assert isinstance(direction.iloc[0, 1], np.bool_)
    assert direction.iloc[4, 1] == True


def test_adjusted_residual1():
    adj_residual = chi_squared_tester.adjusted_residual(
        observed_crosstab=crosstable.copy(deep=True),
        expected_crosstab=chi_squared_tester.expected_crosstab(crosstab_input=crosstable.copy(deep=True)), i=0, j=0
    )
    assert isinstance(adj_residual, np.float64)
    assert round(adj_residual, 2) == -0.66


def test_adjusted_residual2():
    adj_residual = chi_squared_tester.adjusted_residual(
        observed_crosstab=crosstable,
        expected_crosstab=chi_squared_tester.expected_crosstab(crosstab_input=crosstable), i=1, j=0
    )
    assert isinstance(adj_residual, np.float64)
    assert round(adj_residual, 2) == 0.91


def test_chi2_post_hoc_test1():
    stat, p, dof, exp = chi_squared_tester.chi2_stats(crosstab_input=crosstable)
    expected = chi_squared_tester.expected_crosstab(crosstab_input=crosstable)
    values = chi_squared_tester.chi2_post_hoc_test(p_val=p, crosstab_input=crosstable, expected_crosstab=expected)
    assert isinstance(values, pd.DataFrame)
    assert n_clusters == values.shape[1]
    assert isinstance(values.iloc[0, 1], str)
    assert values.iloc[0, 1] == 'neu'
    assert values.iloc[4, 3] == 'neg'
    assert values.iloc[3, 1] == 'pos'


def test_extract_deliver_stats_df():
    chi_squared_tester.extract_deliver_stats_df()
    cluster = chi_squared_tester.deliver_pg_stats
    feature_dict = {'not selected': 'Not Selected'}
    cluster = cluster.replace(feature_dict)
    assert isinstance(cluster, pd.DataFrame)
    assert 'Not Selected' not in cluster['targeting_seg'].unique()


def test_seg_discover_stats_df():
    categories = clustered_data['cluster'].unique()
    chi_squared_tester.extract_deliver_stats_df()
    for category in categories:
        cluster = chi_squared_tester.seg_discover_stats_df(seg_name=category)
        assert isinstance(cluster, pd.DataFrame)

# todo: test json_variables_results
# todo: test json_cluster_results
