library(testthat)
library(jsonlite)
library(arrow)

test_data <- read_parquet('tests/data_for_tests/kenzup_data.parquet')
test_cols_to_include <- fromJSON('tests/data_for_tests/kenzup_cols.json')


test_output <- lca(data = test_data, vars_to_include_json = test_cols_to_include)

previous_output <- fromJSON('tests/data_for_tests/kenzup_lca_output.json')

test_that("LCA code", {
  expect_error(lca())
  expect_equal(length(unlist(test_data[1])), nrow(test_output$data))
  expect_equal(length(unlist(test_data[1])), length(test_output$class_labels))
  expect_equal(length(unlist(test_data[1])), length(previous_output$class_labels))
})
