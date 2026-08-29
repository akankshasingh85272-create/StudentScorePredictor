# Student Score Predictor

## About the Project

I made this project to predict a student's exam score using different factors related to their studies and daily routine.

For example, the dataset has information about how many hours a student studies, their attendance, previous scores, sleep hours, tutoring sessions, motivation level, and a few other factors.

I wanted to see how Machine Learning can use this kind of information to predict a student's exam score.

## Dataset

I used a student performance dataset containing information about **6,607 students**.

The dataset has **20 columns**, including:

- Hours Studied
- Attendance
- Previous Scores
- Sleep Hours
- Tutoring Sessions
- Motivation Level
- Parental Involvement
- Access to Resources
- Internet Access
- Family Income
- Teacher Quality
- Physical Activity
- School Type
- Gender
- and some other student-related information

The score I am trying to predict is:

`Exam_Score`

## What I Did

I started by loading the dataset using Pandas and checking the data to understand what was inside it.

Then I:

- Checked the columns and dataset size.
- Checked for missing values.
- Cleaned the data.
- Separated the features and the exam score.
- Converted the categorical data into numerical form using One-Hot Encoding.
- Split the data into training and testing sets.
- Trained a Random Forest Regressor.
- Used the trained model to predict exam scores.
- Checked the model's performance using MAE, MSE and R² Score.

## Model

For the current version, I used **Random Forest Regressor**.

I chose Random Forest because the dataset contains different types of information, including both numbers and categories.

## Results

The current model gave me these results:

- **MAE:** 1.09
- **MSE:** 4.67
- **R² Score:** 0.67

The model is giving reasonably good predictions, but I still want to improve it and compare it with other models.

## Tools I Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Project Files

```text
StudentScorePredictor/
│
├── StudentPerformanceFactors.csv
├── main.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes