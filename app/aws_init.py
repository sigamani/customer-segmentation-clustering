import datetime

from main import do_segmentation_and_save_to_s3
from app.utils import find_weight_col
import boto3
import os
import json
from io import StringIO
import s3fs
import traceback

s3_check = s3fs.S3FileSystem()

session = boto3.Session()

boto_session = boto3.Session()
s3_client = boto_session.client('s3')
s3 = session.resource('s3')

indi_bet = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/indibet_bettingexternal_india_q4_2022/indibet_bettingexternal_india_q4_2022_responses/indibet_bettingexternal_india_q4_2022_responses.parquet'
go_city = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/gocity_brandawareness_germany_q1_2022/gocity_brandawareness_germany_q1_2022_responses/gocity_brandawareness_germany_q1_2022_responses.parquet'
kenzup = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/kenzup_digitalwalletambassadors_morocco_q2_2022/kenzup_digitalwalletambassadors_morocco_q2_2022_responses/kenzup_digitalwalletambassadors_morocco_q2_2022_responses.parquet'
raiff = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/raiffeisen_ikes_serbia_q4_2021/raiffeisen_ikes_serbia_q4_2021_responses/raiffeisen_ikes_serbia_q4_2021_responses.parquet'
vacc_hes = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/aa_vaccinationhesitancy_morocco_q1_2022/aa_vaccinationhesitancy_morocco_q1_2022_responses/aa_vaccinationhesitancy_morocco_q1_2022_responses.parquet'
part_dem = 's3://qudo-datascience/data-store/prod_surveys/surveys_cleaned/aa_participatorydemocracy_morocco_q4_2022/aa_participatorydemocracy_morocco_q4_2022_responses/aa_participatorydemocracy_morocco_q4_2022_responses.parquet'
fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_responses/qudo_financialservicesfinal_uk_q1_2023_staging_responses.parquet'
fin_ser_preprocessed = 's3://qudo-datascience/data-store/staging_surveys/surveys_preprocessed/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses.parquet'
us_fin_ser = 's3://qudo-datascience/data-store/staging_surveys/surveys_cleaned/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging_responses/qudo_financialservices_usa_q1_2023_staging_responses.parquet'

vacc_hes_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/vaccine_hes.json'
indi_bet_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/indi_bet.json'
go_city_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/go_city.json'
raiff_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/raiff.json'
part_dem_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/participatory_democracy.json'
kenzup_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/kenzup.json'
fin_ser_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging/2023-02-14_14:07:40/cols.json'
us_fin_ser_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservices_usa_q1_2023_staging/2023-03-07_16:19:20/cols.json'

bank_interest_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_bank_interest/2023-02-15_10:30:28/cols.json'
borrowing_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_borrowing/2023-02-15_10:50:33/cols.json'
financial_attitudes_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_financial_attitudes/2023-02-15_11:07:54/cols.json'
financial_behaviour_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_financial_behaviours/2023-02-15_11:24:30/cols.json'
financial_needs_col = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_financial_needs/2023-02-15_11:43:46/cols.json'
bank_switching_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_bank_switching/2023-02-15_11:59:57/cols.json'
inmarket_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_inmarket/2023-02-15_12:14:43/cols.json'
retail_investors_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_retail_investors/2023-02-15_12:35:17/cols.json'
savers_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging_savers/2023-02-15_12:53:02/cols.json'

df_dict = {
    # 'qudo_financialservicesfinal_uk_q1_2023_preprocessed': [fin_ser_preprocessed, None],
    'qudo_financialservicesfinal_uk_q1_2023_staging': [fin_ser, fin_ser_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_bank_interest': [fin_ser, bank_interest_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_borrowing': [fin_ser, borrowing_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_attitudes': [fin_ser, financial_attitudes_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_behaviours': [fin_ser, financial_behaviour_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_needs': [fin_ser, financial_needs_col],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_bank_switching': [fin_ser, bank_switching_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_inmarket': [fin_ser, inmarket_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_retail_investors': [fin_ser, retail_investors_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_savers': [fin_ser, savers_cols],
    # 'qudo_financialservices_usa_q1_2023_staging': [us_fin_ser, us_fin_ser_cols]
    # 'indi_bet': [indi_bet, indi_bet_cols],
    # 'go_city': [go_city, go_city_cols],
    # 'kenzup': [kenzup, kenzup_cols],
    # 'raiff': [raiff, raiff_cols],
    # 'vaccine': [vacc_hes, vacc_hes_cols],
    # 'participatory democracy': [part_dem, part_dem_cols]
}

pipeline_env = os.getenv('PIPELINE_ENV', 'staging')

bucket = 'qudo-datascience'
collected_surveys_key = f"data-store/codebuild-resources/{pipeline_env}/processed_surveys_log/collected_surveys.json"

if s3_check.exists(f's3://{bucket}/{collected_surveys_key}'):
    file = s3_client.get_object(Bucket=bucket, Key=collected_surveys_key)
    collected_surveys = json.load(file['Body'])
else:
    collected_surveys = []
    print("no log file available - stopping")

columns_list_key = f"data-store/codebuild-resources/{pipeline_env}/kraken/feature_columns.json"
columns_file = s3_client.get_object(Bucket=bucket, Key=columns_list_key)
columns = json.load(columns_file['Body'])

faulty_list = []
for i in collected_surveys:
    if "kraken" in str(i["processed_by"]):
        continue

    survey_name = i['title']

    file = f's3://{bucket}/data-store/surveys/{pipeline_env}/responses_cleaned/{survey_name}/{survey_name}.parquet'

    PROJECT_NAME = "_".join(i['title'].split("_")[:-1])
    essential_columns_key = f"data-store/codebuild-resources/essential_columns/essentialcolumns_{PROJECT_NAME}.json"
    s3_check = s3fs.S3FileSystem(use_listings_cache=False)
    if s3_check.exists(f's3://qudo-datascience/{essential_columns_key}'):
        essential_columns_file = s3_client.get_object(Bucket=bucket, Key=essential_columns_key)
        essential_columns = json.load(essential_columns_file['Body'])
        WEIGHT_COLUMN = find_weight_col(data=file,
                                        essential_columns=essential_columns)
    else:
        WEIGHT_COLUMN = None

    try:
        if i['title'] == "qudo_financialservicesfinal_uk_q1_2023_staging":
            cols = fin_ser_cols
        elif i['title'] ==  "qudo_financialservices_usa_q1_2023_staging":
            cols = us_fin_ser_cols
        else:
            cols = columns[i['title']]
    except KeyError:
        continue

    try:
        do_segmentation_and_save_to_s3(i['title'], file, cols, environ=pipeline_env, weight_column=WEIGHT_COLUMN)
        i["processed_by"].append("kraken")

        json_buffer = StringIO()
        json.dump(collected_surveys, json_buffer, indent=2)
        s3.Object(bucket, collected_surveys_key).put(Body=json_buffer.getvalue())

    except Exception as e:
        err_mess = traceback.format_exc().replace("\n", "")
        faulty_list.append({'id': i['id'], 'title': i['title'], 'modified_on': i['modified_on'], 'Error': err_mess})

        json_buffer = StringIO()
        json.dump(faulty_list, json_buffer, indent=2)

        s3.Object(bucket,
                  f"data-store/codebuild-resources/{pipeline_env}/kraken/logs/faulty_surveys_{datetime.datetime.now()}.json").put(
            Body=json_buffer.getvalue())
        s3.Object(bucket,
                  f"data-store/codebuild-resources/{pipeline_env}/kraken/logs/faulty_surveys_latest.json").put(
            Body=json_buffer.getvalue())
