# Video Game Ratings and Playtime: Supervised Machine Learning
# HarvardX Data Science Capstone | November 2026
# Task: Predict log(1 + recorded playtime) from game ratings and engagement.
# Source: https://www.kaggle.com/datasets/shivamvadalia27/video-games
# Run from the Video_Games_Capstone RStudio project directory.
# Required input: data/games.csv
# Install packages once, if necessary: install.packages(c("tidyverse", "randomForest"))

library(tidyverse)
library(randomForest)

# 1. Import and prepare data -------------------------------------------------
if (!file.exists("data/games.csv")) {
  stop("Cannot find data/games.csv. Open the capstone RStudio project first.")
}
games <- read_csv("data/games.csv", show_col_types = FALSE)
games_clean <- distinct(games)
games_model <- games_clean %>%
  filter(!is.na(rating), rating > 0, !is.na(playtime),
         !is.na(added), !is.na(ratings_count))

cat("Original rows:", nrow(games), "\n")
cat("Duplicate rows removed:", nrow(games) - nrow(games_clean), "\n")
cat("Modeling rows:", nrow(games_model), "\n")

# 2. Reproducible 80/20 train/test split ------------------------------------
set.seed(123)
train_index <- sample(seq_len(nrow(games_model)),
                      size = floor(0.8 * nrow(games_model)))
train_data <- games_model[train_index, ]
test_data <- games_model[-train_index, ]

prepare_features <- function(df) {
  df %>% mutate(
    log_playtime = log1p(playtime),
    log_added = log1p(added),
    log_ratings_count = log1p(ratings_count)
  )
}
train_data <- prepare_features(train_data)
test_data <- prepare_features(test_data)

# 3. Evaluation functions: errors on log(1 + playtime) scale ----------------
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
mae <- function(actual, predicted) mean(abs(actual - predicted))
evaluate <- function(model_name, predictions, actual = test_data$log_playtime) {
  tibble(Model = model_name,
         RMSE = rmse(actual, predictions),
         MAE = mae(actual, predictions))
}

# 4. Baseline: training mean -------------------------------------------------
baseline_predictions <- rep(mean(train_data$log_playtime), nrow(test_data))
baseline_results <- evaluate("Baseline", baseline_predictions)

# 5. Supervised model 1: rating-only linear regression ----------------------
lm_model <- lm(log_playtime ~ rating, data = train_data)
lm_predictions <- predict(lm_model, newdata = test_data)
lm_results <- evaluate("Linear Regression - Rating Only", lm_predictions)

# 6. Supervised model 2: rating-only random forest --------------------------
set.seed(123)
rf_model <- randomForest(log_playtime ~ rating, data = train_data,
                         ntree = 500, nodesize = 20)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_results <- evaluate("Random Forest - Rating Only", rf_predictions)

# 7. Supervised model 3: multiple linear regression -------------------------
model_formula <- log_playtime ~ rating + log_added + log_ratings_count
lm_multiple <- lm(model_formula, data = train_data)
lm_multiple_predictions <- predict(lm_multiple, newdata = test_data)
lm_multiple_results <- evaluate("Multiple Linear Regression", lm_multiple_predictions)

# 8. Supervised model 4: multiple-predictor random forest -------------------
set.seed(123)
rf_multiple <- randomForest(model_formula, data = train_data,
                            ntree = 500, nodesize = 20, importance = TRUE)
rf_multiple_predictions <- predict(rf_multiple, newdata = test_data)
rf_multiple_results <- evaluate("Random Forest - Multiple Predictors",
                                rf_multiple_predictions)

# 9. Compare held-out test performance --------------------------------------
model_comparison <- bind_rows(baseline_results, lm_results, rf_results,
                              lm_multiple_results, rf_multiple_results)
cat("\nHeld-out test performance (lower is better):\n")
print(model_comparison, n = Inf)
cat("\nBest model by test RMSE:", model_comparison$Model[which.min(model_comparison$RMSE)], "\n")
cat("\nMultiple linear regression summary:\n")
print(summary(lm_multiple))
cat("\nRandom forest variable importance:\n")
print(importance(rf_multiple))

# 10. Five-fold cross-validation on TRAINING data only ----------------------
set.seed(123)
k <- 5
fold_id <- sample(rep(seq_len(k), length.out = nrow(train_data)))
cv_lm_rmse <- numeric(k)
cv_rf_rmse <- numeric(k)
for (i in seq_len(k)) {
  cv_train <- train_data[fold_id != i, ]
  cv_valid <- train_data[fold_id == i, ]
  cv_lm <- lm(model_formula, data = cv_train)
  cv_lm_rmse[i] <- rmse(cv_valid$log_playtime,
                       predict(cv_lm, newdata = cv_valid))
  cv_rf <- randomForest(model_formula, data = cv_train,
                        ntree = 500, nodesize = 20)
  cv_rf_rmse[i] <- rmse(cv_valid$log_playtime,
                       predict(cv_rf, newdata = cv_valid))
}
cv_results <- tibble(
  Model = c("Multiple Linear Regression", "Random Forest - Multiple Predictors"),
  Mean_RMSE = c(mean(cv_lm_rmse), mean(cv_rf_rmse)),
  SD_RMSE = c(sd(cv_lm_rmse), sd(cv_rf_rmse))
)
cat("\nFive-fold cross-validation (training data only):\n")
print(cv_results)

# 11. Sensitivity check: exclude high-playtime TEST cases for evaluation ----
# This is outcome-selected and is NOT a replacement for full test performance.
playtime_threshold <- as.numeric(quantile(train_data$playtime, 0.99))
non_extreme <- test_data$playtime <= playtime_threshold
sensitivity_results <- tibble(
  Model = c("Multiple Linear Regression", "Random Forest - Multiple Predictors"),
  RMSE = c(rmse(test_data$log_playtime[non_extreme],
                lm_multiple_predictions[non_extreme]),
           rmse(test_data$log_playtime[non_extreme],
                rf_multiple_predictions[non_extreme])),
  MAE = c(mae(test_data$log_playtime[non_extreme],
              lm_multiple_predictions[non_extreme]),
          mae(test_data$log_playtime[non_extreme],
              rf_multiple_predictions[non_extreme]))
)
cat("\n99th-percentile training playtime threshold:", playtime_threshold, "\n")
cat("Test games above threshold:", sum(!non_extreme), "\n")
cat("Sensitivity analysis:\n")
print(sensitivity_results)

# 12. Optional visualization of test errors ---------------------------------
comparison_plot <- model_comparison %>%
  pivot_longer(c(RMSE, MAE), names_to = "Metric", values_to = "Error") %>%
  ggplot(aes(x = reorder(Model, Error), y = Error, fill = Metric)) +
  geom_col(position = "dodge") + coord_flip() + theme_minimal() +
  labs(title = "Supervised Model Performance on Held-Out Test Data",
       x = "Model", y = "Prediction error (log playtime scale)")
print(comparison_plot)

# Observational associations do not establish causality.
# The zero-rating exclusion may limit generalizability.
