import numpy as np
import pandas as pd
import boto3
import json

import pytest

from ..app.clustering import Clusterings
from ..app.main import remove_na_strings_and_floats, remove_na_strings

fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'
fin_ser = pd.read_parquet(fin_ser)
fin_ser = fin_ser.apply(remove_na_strings)

new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in fin_ser.columns]
fin_ser.columns = new_cols
fin_ser.rename(columns={'qudo': 'rules_based'}, inplace=True)

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


preprocessed_uri = 's3://qudo-datascience/data-store/surveys/staging/responses_preprocessed/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'
preprocessed_df = pd.read_parquet(preprocessed_uri)
preprocessed_df = preprocessed_df.apply(remove_na_strings_and_floats)
new_cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in preprocessed_df.columns]
preprocessed_df.columns = new_cols
preprocessed_df.rename(columns={'qudo': 'rules_based'}, inplace=True)
wrong_uri = 'senrglosenrg'


def test_wrong_uri():
    with pytest.raises(AttributeError):
        cluster_obj = Clusterings(wrong_uri, fin_ser_cols, full_data=fin_ser)


def test_instantiation():
    cluster_obj = Clusterings(preprocessed_df, fin_ser_cols, full_data=fin_ser)

