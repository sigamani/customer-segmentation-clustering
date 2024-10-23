import numpy as np
import pandas as pd
import boto3
import json

import pytest

from ..app.clustering import Clusterings
from ..app.main import remove_na_strings_and_floats, remove_na_strings


us_fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'

fin_ser = pd.read_parquet(us_fin_ser)
fin_ser = fin_ser.apply(remove_na_strings)
new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser.columns]
fin_ser.columns = new_cols

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

columns_uri = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservices_usa_q1_2023_staging/2023-03-07_16:19:20/cols.json'
splits = columns_uri.split('/')
bucket_name = splits[2]
key_name = "/".join(splits[3:])
content_obj = s3.Object(bucket_name, key_name)
content = content_obj.get()['Body'].read().decode('utf-8')
fin_ser_cols = json.loads(content)
fin_ser_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser_cols]

wrong_cols = {col: col.replace("qudo", "mudo") for col in fin_ser.columns}
wrong_cols_df = fin_ser.rename(columns=wrong_cols)
wrong_cols_list = [x.replace("qudo", "mudo") for x in fin_ser.columns if "qudo" in x]


def test_wrong_cols():
    cluster_obj = Clusterings(wrong_cols_df, wrong_cols_list)
    rules_based = []
    recommended = cluster_obj.rules_based(q_codes=rules_based)
    assert isinstance(recommended, list)
    assert len(recommended) == 0


def test_recommended():
    rules_based = [x for x in fin_ser.columns if 'qudo' in x or 'cluster' in x]
    cluster_obj = Clusterings(fin_ser, fin_ser_cols)
    recommended = cluster_obj.rules_based(q_codes=rules_based[1:3])
    assert isinstance(recommended, list)
    assert len(recommended) == 2
    assert isinstance(recommended[0], dict)
    assert isinstance(recommended[0]['model'], str)
    assert isinstance(recommended[0]['labels'], np.ndarray)
    assert isinstance(recommended[0]['metrics'], dict)
    assert isinstance(recommended[0]['inference_data'], tuple)

