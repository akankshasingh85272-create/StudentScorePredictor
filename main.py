# Import the Pandas Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Display the First 5 rows
print(df.head())



print("\nDataset Shape: ")
print(df.shape)

print("\nColumn Names: ")
print(df.columns)

print("\nDataset Information: ")
df.info()

print("\nStatistical Summary: ")
print(df.describe())


print("\nMissing Values: ")
print(df.isnull().sum())


print("\nOriginal DataFrame:")
print(df.head())

new_df = df.copy()

categorical_columns = new_df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    new_df[column] = new_df[column].fillna(new_df[column].mode()[0])

print("\nAfter handling missing values:")
print(new_df.isnull().sum()) 

X = new_df.drop(columns=['Exam_Score'])
y = new_df['Exam_Score']

print("\nFeatures (X):")
print(X.head())

print("\n-------------------\n")

print("\nTarget (Y):")
print(y.head())

X = new_df.drop(columns=['Exam_Score'])
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('numerical', 'passthrough', numerical_columns)
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX_train:")
print(X_train.head())


print("\nX_test:")
print(X_test.head())

linear_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_prediction)
linear_mse = mean_squared_error(y_test, linear_prediction)
linear_r2 = r2_score(y_test, linear_prediction)

print("\n--- Linear Regression Results ---")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("R2 Score:", linear_r2)


model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Predicted Values:")
print(prediction)

print("Actual Values:")
print(y_test.values)

print("Rounded Prediction:", round(prediction[0]))
print("Prediction (2 decimals):", round(prediction[0], 2))


mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print( "Mean Absolute Error:",mae )
print("Mean Squared Error:",mse )
print("R2 Score:", r2 )


