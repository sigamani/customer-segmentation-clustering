import json

import pandas as pd
import pytest

from ..app.clustering import Clusterings
from ..app.metrics import get_social_presence

indi_bet = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/indibet_bettingexternal_india_q4_2022/indibet_bettingexternal_india_q4_2022_responses/indibet_bettingexternal_india_q4_2022_responses.parquet')

with open('local/data/cluster_cols/indibet.json', 'r') as f:
    indi_cols = json.load(f)

indi_cols = [x.lower() for x in indi_cols]
indi_cols = [col for col in indi_bet.columns for x in indi_cols if x in col]
indi_cluster_obj = Clusterings(indi_bet, indi_cols)
indi_cluster_result = indi_cluster_obj.kmeans_and_pca_clustering()
labels = indi_cluster_result.get('labels')


def test_wrong_argument_entered():
    with pytest.raises(Exception) as e_info:
        get_social_presence(indi_bet, labels, 'FB')


def test_object_is_dict():
    test_obj = get_social_presence(indi_bet, labels, 'Facebook')
    assert isinstance(test_obj, dict)
