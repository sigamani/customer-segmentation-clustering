from main import do_segmentation_and_save_to_s3
from utils import find_weight_col
import s3fs
import boto3
import json
import pandas as pd

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
part_dem = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/aa_participatorydemocracy_morocco_q4_2022_staging/aa_participatorydemocracy_morocco_q4_2022_staging.parquet'
fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
fin_ser_preprocessed = 's3://qudo-datascience/data-store/staging_surveys/surveys_preprocessed/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses/qudo_financialservicesfinal_uk_q1_2023_staging_preprocessed_responses.parquet'
fin_ser_new = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
fin_ser_preprocessed_new = 's3://qudo-datascience/data-store/surveys/staging/responses_preprocessed/qudo_financialservicesfinal_uk_q1_2023_staging/qudo_financialservicesfinal_uk_q1_2023_staging.parquet'
us_fin_ser = 's3://qudo-datascience/data-store/surveys/staging/responses_cleaned/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'
us_fin_ser_preprocessed = 's3://qudo-datascience/data-store/surveys/staging/responses_preprocessed/qudo_financialservices_usa_q1_2023_staging/qudo_financialservices_usa_q1_2023_staging.parquet'
part_dem_preprocessed = 's3://qudo-datascience/data-store/surveys/staging/responses_preprocessed/aa_participatorydemocracy_morocco_q4_2022_staging/aa_participatorydemocracy_morocco_q4_2022_staging.parquet'

vacc_hes_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/vaccine_hes.json'
indi_bet_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/indi_bet.json'
go_city_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/go_city.json'
raiff_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/raiff.json'
part_dem_cols = 's3://qudo-datascience/data-store/lachsesis/aa_participatorydemocracy_morocco_q4_2022_staging/ml/cols.json'
kenzup_cols = 's3://qudo-datascience/data-store/kraken_outputs/test/feature_columns/curated/kenzup.json'
fin_ser_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_staging/2023-02-14_09:53:53/cols.json'
us_fin_ser_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservices_usa_q1_2023_staging/2023-03-07_16:19:20/cols.json'
fin_ser_preprocessed_cols = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/test/qudo_financialservicesfinal_uk_q1_2023_preprocessed/2023-02-21_17:08:09/cols.json'
new_fin_cols_feature_selected = 's3://qudo-datascience/data-store/kraken_outputs/feature_columns/staging/qudo_financialservicesfinal_uk_q1_2023/2023-03-09_10:35:06/cols.json'


df_dict = {
     'qudo_financialservicesfinal_uk_q1_2023_staging': [fin_ser_new, fin_ser_preprocessed_cols, fin_ser_preprocessed_new],
    #'qudo_financialservices_usa_q1_2023_staging': [us_fin_ser, us_fin_ser_cols, us_fin_ser_preprocessed],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging': [fin_ser, fin_ser_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_bank_interest': [fin_ser, bank_interest_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_borrowing': [fin_ser, borrowing_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_attitudes': [fin_ser, financial_attitudes_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_behaviours': [fin_ser, financial_behaviour_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_financial_needs': [fin_ser, financial_needs_col],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_bank_switching': [fin_ser, bank_switching_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_inmarket': [fin_ser, inmarket_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_retail_investors': [fin_ser, retail_investors_cols],
    # 'qudo_financialservicesfinal_uk_q1_2023_staging_savers': [fin_ser, savers_cols],
    # 'qudo_financialservices_usa_q1_2023_staging': [us_fin_ser, us_fin_ser_cols],
    # 'indi_bet': [indi_bet, indi_bet_cols],
    # 'go_city': [go_city, go_city_cols],
    # 'kenzup': [kenzup, kenzup_cols],
    # 'raiff': [raiff, raiff_cols],
    # 'vaccine': [vacc_hes, vacc_hes_cols],
    # 'participatory democracy': [part_dem, part_dem_cols]
    # 'aa_participatorydemocracy_morocco_q4_2022_staging': [part_dem, part_dem_cols]
}

# qudo_fin_ser_segmentations = pd.read_csv('../local/data/Financial Services Q1 2023 Alchemer Upload - Financial Services Q1 2023 Alchemer Upload.csv')
for survey_name, data in df_dict.items():

    bucket = 'qudo-datascience'
    file = pd.read_parquet(data[0])
    #file["precompletion_weight"] = 1.1 just for testing with weights

    PROJECT_NAME = "_".join(survey_name.split("_")[:-1])
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
        do_segmentation_and_save_to_s3(survey_name, data[0], data[1], preprocessed=data[2], environ='test',
                                       weight_column=WEIGHT_COLUMN)
    except IndexError:
        print('this is running')
        do_segmentation_and_save_to_s3(survey_name, data[0], data[1], environ='test',  weight_column=WEIGHT_COLUMN)
