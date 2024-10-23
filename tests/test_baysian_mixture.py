import numpy as np
import pandas as pd
import boto3
import json

from sklearn.mixture import BayesianGaussianMixture
from ..app.clustering import Clusterings
from ..app.feature_selection.feature_selection import rank_features

fin_ser_demo = pd.read_parquet(
    's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet')
new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser_demo.columns]
fin_ser_demo.columns = new_cols
fin_ser_demo.rename(columns={'qudo': 'rules_based'}, inplace=True)



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

try:
    fin_ser_demo.drop('qudo', axis=1, inplace=True)
except KeyError:
    pass
try:
    fin_ser_demo.drop('cluster', axis=1, inplace=True)
except KeyError:
    pass


def remove_na_strings(col):
    if col.dtypes == 'string':
        col = col.fillna('None')
    else:
        col = col.fillna(-999)
    return col


fin_ser_demo = fin_ser_demo.apply(remove_na_strings)
cluster_obj = Clusterings(fin_ser_demo, fin_ser_cols)

bmm = cluster_obj.bmm()


def test_bmm():
    assert len(bmm) == 4
    assert isinstance(bmm['model'], BayesianGaussianMixture)
    assert isinstance(bmm['labels'], pd.Series)
    assert isinstance(bmm['labels'][0], np.int64)
    assert isinstance(bmm['metrics'], dict)
    assert isinstance(bmm['inference_data'], tuple)


def test_baysian_mixture_modelling():
    baysian_mixture_modelling = cluster_obj.baysian_mixture_modelling(
        cluster_obj.data_encoded[cluster_obj.cluster_vars], 3)
    assert len(baysian_mixture_modelling) == 3
    assert isinstance(baysian_mixture_modelling, tuple)
    assert isinstance(baysian_mixture_modelling[1], BayesianGaussianMixture)
    assert isinstance(baysian_mixture_modelling[0], pd.Series)
    assert isinstance(baysian_mixture_modelling[0][0], np.int64)
    assert isinstance(baysian_mixture_modelling[2], dict)
