import argparse
import boto3
import datetime
import json
import logging
import random

import numpy as np
import pandas as pd
import pickle
from itertools import chain
try:
    try:
        from app.feature_selection.feature_selection import rank_features
    except ModuleNotFoundError:
        from feature_selection.feature_selection import rank_features
    from clustering import Clusterings
except ModuleNotFoundError:
    from ..app.feature_selection.feature_selection import rank_features
    from ..app.clustering import Clusterings


s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


def remove_na_strings(col):
    if col.dtypes == 'string':
        col = col.fillna('not selected')
    return col


def remove_na_strings_and_floats(col):
    if col.dtypes == 'string':
        col = col.fillna('not selected')
    else:
        col = col.fillna(-999)
    return col


def map_to_option_title(survey_name, r_responses_df):
    preprocessed_questions_uri = f's3://qudo-datascience/data-store/surveys/staging/questions_preprocessed/{survey_name}/{survey_name}.parquet'
    preprocessed_questions_df = pd.read_parquet(preprocessed_questions_uri)
    mismatch_df = preprocessed_questions_df.loc[~(preprocessed_questions_df['option_text'] == preprocessed_questions_df['option_value'])][['varname', 'option_text', 'option_value']].drop_duplicates().copy()
    mismatch_questions = mismatch_df['varname'].unique()

    responses_df = r_responses_df.copy()
    responses_columns = responses_df.columns

    for mismatch_col in responses_columns:
        if mismatch_col in mismatch_questions:
            mismatch_mappings = mismatch_df[mismatch_df['varname'] == mismatch_col].copy()
            mismatch_mappings_dict = dict(zip(mismatch_mappings['option_value'], mismatch_mappings['option_text']))
            try:
                responses_df[mismatch_col] = responses_df[mismatch_col].apply(lambda x: mismatch_mappings_dict[x] if x in mismatch_mappings_dict.keys() else x)
            except:
                raise KeyError(f'Mapping is not complete, please check all options for varname {mismatch_col}.')
    return responses_df


def do_segmentation_and_save_to_s3(survey_name, data_uri, columns_uri=None, environ='staging', data_provided=False,
                                   rules_based=None, hierarchical=None, ignore_hierarchical_value=None,
                                   add_manual_seg_columns=None, preprocessed=None, weight_column=None):

    nowish = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
    if preprocessed:
        preprocessed_df = pd.read_parquet(preprocessed)
        preprocessed_df = preprocessed_df.apply(remove_na_strings_and_floats)
    if data_provided:
        df = data_uri
    else:
        df = pd.read_parquet(data_uri)
    df = df.apply(remove_na_strings)

    df = map_to_option_title(survey_name, df)

    # df["precompletion_weight"] = [random.random() for item in range(df.shape[0])] # just for testing weights

    if isinstance(add_manual_seg_columns, str):
        add_manual_seg_columns = pd.read_csv(add_manual_seg_columns)

    if isinstance(add_manual_seg_columns, pd.DataFrame):
        add_manual_seg_columns.columns = [x.lower() for x in add_manual_seg_columns.columns]
        cols_to_retain = ['vrid', 'id']
        for col in add_manual_seg_columns:
            if 'cint' in col:
                cols_to_retain.append(col)
        add_manual_seg_columns.columns = [f'qudo_{x}' if x not in cols_to_retain else x for x in add_manual_seg_columns.columns]
        df = pd.merge(df, add_manual_seg_columns, left_on='cint_id', right_on='id', how='left')
    ### Column Selection
    if columns_uri and not data_provided:
        splits = columns_uri.split('/')
        bucket_name = splits[2]
        key_name = "/".join(splits[3:])
        content_obj = s3.Object(bucket_name, key_name)
        content = content_obj.get()['Body'].read().decode('utf-8')
        cols = json.loads(content)
    elif data_provided:
        col_fragments = [x.lower() for x in columns_uri]
        cols = []
        for col in col_fragments:
            cols.append([x for x in df.columns if col in x])
        cols = list(chain(*cols))
        cols_json = json.dumps(cols)
        cols_s3_uri = f'data-store/kraken_outputs/feature_columns/{environ}/{survey_name}/{nowish}/cols.json'
        s3.Object('qudo-datascience', cols_s3_uri).put(Body=cols_json)
    else:
        if preprocessed:
            cols = rank_features(preprocessed_df, 40, 5, 1, 4)
            print(cols)
        else:
            cols = rank_features(df, 40, 5, 1, 4)
        try:
            cols_json = json.dumps(cols)
        except:
            try:
                cols_json = json.dumps(cols.to_json())
            except:
                cols_json = list(cols)
        cols_s3_uri = f'data-store/kraken_outputs/feature_columns/{environ}/{survey_name}/{nowish}/cols.json'
        s3.Object('qudo-datascience', cols_s3_uri).put(Body=cols_json)

    ### Segmentation
    if len([x for x in cols if '_tgt' in x]) == 0:
        cols = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in cols]
        df.columns = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in df.columns]
        if preprocessed:
            cols = [col.replace('_numeric', '') if '_numeric' in col else col for col in cols]
            preprocessed_df.columns = [col.replace('_numeric', '') if '_numeric' in col else col for col in
                                       preprocessed_df.columns]
            preprocessed_df.columns = [col + '_tgt' if '_fb' in col or '_gg' in col else col for col in
                                       preprocessed_df.columns]

    if preprocessed:
        segmentation_obj = Clusterings(preprocessed_df, cols, hierarchical=hierarchical,
                                       ignore_hierarchical_value=ignore_hierarchical_value, full_data=df,
                                       weight_col=weight_column)
    else:
        segmentation_obj = Clusterings(df, cols, hierarchical=hierarchical,
                                       ignore_hierarchical_value=ignore_hierarchical_value,
                                       weight_col=weight_column)
    if rules_based:
        if isinstance(rules_based, list):
            pass
        else:
            rules_based = [rules_based]
    else:
        rules_based = [x for x in df.columns if 'qudo' in x or 'cluster' in x]
        try:
            rules_based.remove('qudo_weight_precompletes')
        except ValueError:
            pass
    # if rules_based:
    #     segmentations = segmentation_obj.run_all_segmentations(q_code=rules_based)
    # else:
    #     segmentations = segmentation_obj.run_all_segmentations()

    segmentations = {'kmeans': segmentation_obj.kmeans_and_pca_clustering()} #segmentation_obj.kmeans_and_pca_clustering()
    metrics_df = pd.DataFrame(x['metrics'] for x in segmentations.values())
    metrics_s3_uri = f's3://qudo-datascience/data-store/kraken_outputs/{environ}/{survey_name}/{nowish}/metrics.csv'
    metrics_df.to_csv(metrics_s3_uri, index=False)

    for seg_name, seg_data in segmentations.items():
        seg_data_s3_uri = f'data-store/kraken_outputs/{environ}/{survey_name}/{nowish}/{seg_name}.pickle'
        pickle_byte_obj = pickle.dumps(seg_data)
        s3.Object('qudo-datascience', seg_data_s3_uri).put(Body=pickle_byte_obj)
    return f'data-store/kraken_outputs/{environ}/{survey_name}/{nowish}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNLEASH THE KRAKEN')
    parser.add_argument("--surveyname", required=True)
    parser.add_argument("--datas3uri", required=True)
    parser.add_argument("--collists3uri")
    parser.add_argument("--environment")

    args = parser.parse_args()
    data_s3 = args.datas3uri
    col_s3 = args.collists3uri
    survey_name = args.surveyname
    environ = args.environment
    if not environ:
        environ = 'staging'
    saved_uri = do_segmentation_and_save_to_s3(survey_name, data_s3, col_s3, environ=environ)
    logging.log(1, saved_uri)


