import numpy as np
import pandas as pd
import boto3
import json
from ..app.clustering import Clusterings

fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
fin_ser = pd.read_parquet(fin_ser)

new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser.columns]
fin_ser.columns = new_cols
fin_ser.rename(columns={'qudo': 'rules_based'}, inplace=True)

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

columns_uri = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging/2023-02-14_09:53:53/cols.json'
splits = columns_uri.split('/')
bucket_name = splits[2]
key_name = "/".join(splits[3:])
content_obj = s3.Object(bucket_name, key_name)
content = content_obj.get()['Body'].read().decode('utf-8')
fin_ser_cols = json.loads(content)
fin_ser_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser_cols]


def remove_na_strings(col):
    if col.dtypes == 'string':
        col = col.fillna('None')
    else:
        col = col.fillna(-999)
    return col


fin_ser = fin_ser.apply(remove_na_strings)
cluster_obj = Clusterings(fin_ser, fin_ser_cols)

rules_based = cluster_obj.rules_based(['qudo'])


def test_rules_based():
    assert len(rules_based) == 4
    assert isinstance(rules_based['model'], str)
    assert rules_based['model'] == 'Rules-Based'
    assert isinstance(rules_based['metrics'], dict)
    assert isinstance(rules_based['labels'], np.ndarray)
    assert isinstance(rules_based['labels'][0], np.int64)
    assert isinstance(rules_based['inference_data'], tuple)
