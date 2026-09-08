import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("StudentPerformanceFactors.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())


# --------------------------------
# 2. Feature Engineering
# --------------------------------

new_df = df.copy()

new_df["Study_Engagement"] = (
    new_df["Hours_Studied"]
    * new_df["Attendance"]
    / 100
)

print("\n--- Study Engagement ---")
print(
    new_df[
        ["Hours_Studied", "Attendance", "Study_Engagement"]
    ].head()
)


# --------------------------------
# 3. Separate Features and Target
# --------------------------------

X = new_df.drop(columns=["Exam_Score"])
y = new_df["Exam_Score"]


# --------------------------------
# 4. Identify Column Types
# --------------------------------

categorical_columns = X.select_dtypes(
    include=["str"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["str"]
).columns


# --------------------------------
# 5. Preprocessing
# --------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_columns
        ),
        (
            "numerical",
            numerical_transformer,
            numerical_columns
        )
    ]
)


# --------------------------------
# 6. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 7. Linear Regression Model
# --------------------------------

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_mse = mean_squared_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)

print("\n--- Linear Regression Results ---")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("R2 Score:", linear_r2)


# --------------------------------
# 8. Save Best Model
# --------------------------------

joblib.dump(
    linear_model,
    "student_score_model.pkl"
)

print("\nLinear Regression model saved successfully!")


# --------------------------------
# 9. Random Forest Model
# --------------------------------

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_prediction = random_forest_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_mse = mean_squared_error(
    y_test,
    rf_prediction
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)

print("\n--- Random Forest Results ---")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)


# --------------------------------
# 10. Model Comparison
# --------------------------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "MSE": [
        linear_mse,
        rf_mse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

print("\n--- Model Comparison ---")
print(results)

best_model = results.loc[
    results["R2 Score"].idxmax()
]

print("\n--- Best Model ---")
print("Model:", best_model["Model"])
print("R2 Score:", best_model["R2 Score"])


# --------------------------------
# 11. Hyperparameter Tuning
# --------------------------------

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [
        None,
        10,
        20
    ],
    "regressor__min_samples_split": [
        2,
        5
    ]
}

grid_search = GridSearchCV(
    random_forest_model,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- Best Random Forest Parameters ---")
print(grid_search.best_params_)

print("\n--- Tuned Random Forest Cross-Validation R2 ---")
print(grid_search.best_score_)


# --------------------------------
# 12. Tuned Random Forest Evaluation
# --------------------------------

tuned_model = grid_search.best_estimator_

tuned_prediction = tuned_model.predict(X_test)

tuned_mae = mean_absolute_error(
    y_test,
    tuned_prediction
)

tuned_mse = mean_squared_error(
    y_test,
    tuned_prediction
)

tuned_r2 = r2_score(
    y_test,
    tuned_prediction
)

print("\n--- Tuned Random Forest Test Results ---")
print("MAE:", tuned_mae)
print("MSE:", tuned_mse)
print("R2 Score:", tuned_r2)


# --------------------------------
# 13. Final Model Comparison
# --------------------------------

final_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae,
        tuned_mae
    ],
    "MSE": [
        linear_mse,
        rf_mse,
        tuned_mse
    ],
    "R2 Score": [
        linear_r2,
        rf_r2,
        tuned_r2
    ]
})

print("\n--- Final Model Comparison ---")
print(final_results)

best_model_name = final_results.loc[
    final_results["R2 Score"].idxmax(),
    "Model"
]

print("\n--- Final Best Model ---")
print("Model:", best_model_name)


# --------------------------------
# 14. Feature Importance
# --------------------------------

feature_names = tuned_model[
    "preprocessor"
].get_feature_names_out()

feature_importances = tuned_model[
    "regressor"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Top 10 Important Features ---")
print(importance_df.head(10))


# --------------------------------
# 15. EDA - Exam Score Distribution
# --------------------------------

plt.hist(
    df["Exam_Score"],
    bins=20
)

plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Exam Scores")

plt.show()


# --------------------------------
# 16. EDA - Hours Studied vs Exam Score
# --------------------------------

plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"]
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")

plt.show()


# --------------------------------
# 17. Correlation Analysis
# --------------------------------

correlation = df.select_dtypes(
    include="number"
).corr()

print("\n--- Correlation with Exam Score ---")
print(
    correlation["Exam_Score"]
    .sort_values(ascending=False)
)


# --------------------------------
# 18. Correlation Bar Chart
# --------------------------------

sorted_correlation = (
    correlation["Exam_Score"]
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    sorted_correlation.index,
    sorted_correlation.values
)

plt.xlabel("Correlation")
plt.ylabel("Features")
plt.title("Correlation of Features with Exam Score")

plt.show()


# --------------------------------
# 19. Feature Importance Chart
# --------------------------------

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Feature Importance")

plt.show()