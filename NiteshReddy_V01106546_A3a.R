# Load necessary libraries
library(tidyverse)
library(caret)
library(ROCR)  # For ROC curve and AUC calculation
library(rpart)  # For Decision Tree Classifier

# Set the working directory and import the dataset
setwd("/Users/niteshreddy/Downloads/")  # Replace with your actual directory path
df <- read.csv('Credit Card Defaulter Prediction.csv')

# Data Cleaning and EDA
# Remove spaces in column names
colnames(df)[colnames(df) == "default "] <- "default"

# Check class balance
table(df$default)

# Remove unnecessary column 'ID'
df <- df[, !(names(df) %in% c("ID"))]

# Identify categorical features
cat_features <- df %>% select_if(is.character)

# Data Encoding for Categorical Variables
encoded_num_df <- as.data.frame(lapply(cat_features, as.factor))

# Final data preparation
f_data <- bind_cols(encoded_num_df, df %>% select(-SEX, -EDUCATION, -MARRIAGE, -default))

# Test-Train Split
set.seed(42)  # Set seed for reproducibility
trainIndex <- createDataPartition(f_data$default, p = 0.8, list = FALSE)
train_data <- f_data[trainIndex, ]
test_data <- f_data[-trainIndex, ]

# Scaling the Data
# Scale data to 0-1 range
sc <- preProcess(train_data[, -which(names(train_data) %in% c("default"))], method = c("range"))
sc_train_data <- predict(sc, train_data)
sc_test_data <- predict(sc, test_data)

# Feature Selection using Random Forest Variable Importance
# Train Random Forest model
rf_model <- train(default ~ ., data = sc_train_data, method = "rf")
var_imp <- varImp(rf_model)

# Select top 15 features based on importance
selected_features <- rownames(var_imp$importance)[1:15]
print(paste("Selected Features:", selected_features))

# Select data with selected features
feature_selection_train <- sc_train_data[, c(selected_features, "default")]
feature_selection_test <- sc_test_data[, c(selected_features, "default")]

# Logistic Regression
# Fit Logistic Regression
logreg <- glm(default ~ ., data = feature_selection_train, family = binomial)

# Predict on test set
y_pred <- ifelse(predict(logreg, newdata = feature_selection_test, type = "response") > 0.5, 1, 0)

# Classification Report
logrepo <- confusionMatrix(data = factor(y_pred), reference = factor(feature_selection_test$default))

# ROC Curve and AUC Value
# Predicted probabilities
y_pred_proba_log <- predict(logreg, newdata = feature_selection_test, type = "response")
pred <- prediction(y_pred_proba_log, feature_selection_test$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

# Decision Tree Classifier
# Train Decision Tree Classifier
dt_classifier <- rpart(default ~ ., data = feature_selection_train, method = "class")

# Predict on test set
y_pred_dt <- predict(dt_classifier, newdata = feature_selection_test, type = "class")

# Classification Report
dtree <- confusionMatrix(data = factor(y_pred_dt), reference = factor(feature_selection_test$default))

# ROC Curve and AUC Value for Decision Tree
y_pred_proba <- predict(dt_classifier, newdata = feature_selection_test)[, 2]
pred_dt <- prediction(y_pred_proba, feature_selection_test$default)
perf_dt <- performance(pred_dt, "tpr", "fpr")
plot(perf_dt, colorize = TRUE)

# Display Confusion Matrices
print(logrepo)
print(dtree)

# Comparison Table
comparison_df <- rbind(
  as.data.frame(logrepo$byClass),
  as.data.frame(dtree$byClass)
)
comparison_df$model <- c("Logistic Regression", "Decision Tree")
rownames(comparison_df) <- NULL
comparison_df <- comparison_df[, c("model", "Sensitivity", "Specificity", "Precision", "Recall", "F1", "Balanced Accuracy")]

print(comparison_df)

