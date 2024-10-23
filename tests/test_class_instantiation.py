import json

import pandas as pd
import pytest

from ..app.clustering import Clusterings

indi_bet = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/indibet_bettingexternal_india_q4_2022/indibet_bettingexternal_india_q4_2022_responses/indibet_bettingexternal_india_q4_2022_responses.parquet')

with open('local/data/cluster_cols/indibet.json', 'r') as f:
    indi_cols = json.load(f)

indi_cols = [x.lower() for x in indi_cols]
indi_cols = [col for col in indi_bet.columns for x in indi_cols if x in col]

working_obj = Clusterings(indi_bet, indi_cols)


def test_obj_has_attributes():
    assert hasattr(working_obj, 'data_encoded')
    assert hasattr(working_obj, 'num_cores')
    assert hasattr(working_obj, 'gower_matrix')
    assert hasattr(working_obj, 'method_kprototypes_or_kmodes')
    assert hasattr(working_obj, 'opt_clusters')
    assert hasattr(working_obj, 'seeds')


with open('local/data/cluster_cols/go_city.json', 'r') as f:
    go_city_cols = json.load(f)


def test_failing_obj_instantiation():
    with pytest.raises(KeyError):
        mismatched_cols = Clusterings(indi_bet, go_city_cols)
    with pytest.raises(KeyError):
        empty_df = Clusterings(pd.DataFrame(), indi_cols)
