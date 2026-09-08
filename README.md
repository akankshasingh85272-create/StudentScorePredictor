# Student Score Predictor

This is a machine learning project that predicts a student's exam score based on different academic and personal factors.

I built this project to practice the complete machine learning process, from exploring the data to building a model and creating a simple web application.

## About the Project

The project uses student information such as:

- Hours studied
- Attendance
- Previous scores
- Tutoring sessions
- Sleep hours
- Motivation level
- Parental involvement
- Access to resources
- And other student-related factors

The model uses these details to predict the expected exam score.

## What I Did

In this project, I worked on:

- Loading and exploring the dataset
- Checking missing values
- Exploratory Data Analysis (EDA)
- Creating graphs to understand the data
- Feature engineering
- Comparing Linear Regression and Random Forest
- Evaluating the models using MAE, MSE and R²
- Hyperparameter tuning for Random Forest
- Checking feature importance
- Saving the trained model
- Creating a Streamlit web application for predictions

## Model Results

I compared two machine learning models.

| Model | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.45 | 3.25 | 0.77 |
| Random Forest | 1.07 | 4.70 | 0.67 |

Linear Regression performed better on the test data, so I used it as the final model for the prediction application.

## Web Application

The project also includes a simple Streamlit application.

Users can enter student details and get a predicted exam score.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Project Files

- `main.py` - data analysis, model training and evaluation
- `app.py` - Streamlit web application
- `StudentPerformanceFactors.csv` - dataset
- `student_score_model.pkl` - saved trained model
- `requirements.txt` - required Python packages

## How to Run

Install the required packages:

    pip install -r requirements.txt

Run the web application:

    streamlit run app.py

## What I Learned

This project helped me understand the basic machine learning workflow and how a trained model can be connected to a simple web application.

## Future Improvements

I would like to improve the user interface, test the model with more data, and improve the overall prediction system in the future.

## Live Demo

You can try the Student Score Predictor here:

[Open the Live App] (https://studentscorepredictor-gye8lappejh4jsfcoqsynsr.streamlit.app/)

