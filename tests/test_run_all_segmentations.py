import json

import boto3
import pandas as pd

try:
    from app.clustering import Clusterings
except ModuleNotFoundError:
    from ..app.clustering import Clusterings

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

indi_bet = pd.read_parquet(
    's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/indibet_bettingexternal_india_q4_2022/indibet_bettingexternal_india_q4_2022_responses/indibet_bettingexternal_india_q4_2022_responses.parquet')

indi_bet_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/indi_bet.json'

splits = indi_bet_cols.split('/')
bucket_name = splits[2]
key_name = "/".join(splits[3:])
content_obj = s3.Object(bucket_name, key_name)
content = content_obj.get()['Body'].read().decode('utf-8')
cols = json.loads(content)
indi_bet_with_recommended_cols = cols.copy()
indi_bet_with_recommended = indi_bet.copy(deep=True)
indi_bet.drop('cluster', axis=1, inplace=True)


def test_run_all_segmentations_no_recommended():
    indi_clusterings = Clusterings(indi_bet, cols)
    run_all_segmentations = indi_clusterings.run_all_segmentations()
    assert isinstance(run_all_segmentations, dict)
    assert len(run_all_segmentations) == 6
    for seg in run_all_segmentations.values():
        assert isinstance(seg, dict)


def test_run_all_segmentations_recommended():
    indi_clusterings = Clusterings(indi_bet_with_recommended, indi_bet_with_recommended_cols)
    run_all_segmentations = indi_clusterings.run_all_segmentations(q_code='cluster')
    assert isinstance(run_all_segmentations, dict)
    assert len(run_all_segmentations) == 7
    for seg in run_all_segmentations.values():
        assert isinstance(seg, dict)
