library(testthat)
library(jsonlite)
library(arrow)

test_data <- read_parquet('tests/data_for_tests/vacc_data_for_mm.parquet')
test_cols_to_include <- fromJSON('tests/data_for_tests/vacc_columns.json')
test_data_types <- fromJSON('tests/data_for_tests/vacc_datatypes.json')

test_output <- mixture_modelling_lca(data = test_data, vars_to_include_json = test_cols_to_include,
                                     data_types_for_vars = test_data_types)

previous_output <- fromJSON('tests/data_for_tests/vacc_mm_output.json')
test_that("Mixture Model code", {
  expect_error(mixture_modelling_lca())
  expect_equal(length(unlist(test_data[1])), nrow(test_data))
  expect_equal(length(unlist(test_data[1])), length(test_output$class_labels))
  expect_equal(length(unlist(test_data[1])), length(previous_output$class_labels))
})