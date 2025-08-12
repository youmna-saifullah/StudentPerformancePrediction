🎯 Student Performance Prediction using Decision Tree
This project predicts whether a student will Pass or Fail based on their academic scores and personal background using a Decision Tree Classifier.
It also provides a Tkinter-based GUI for user interaction and a Decision Tree visualization for better understanding of how the model makes predictions.

📌 Features
Data Preprocessing:
Removes irrelevant columns (race/ethnicity)
Encodes categorical variables into numeric form
Creates a new average_score column
Generates a Pass/Fail target variable
Model Training:
Uses DecisionTreeClassifier from scikit-learn
Splits dataset into training & testing sets
Visualization:
Displays the decision tree for understanding model logic
GUI Interface:
Users can input gender, parental education, lunch type, test preparation, and scores
Click "Predict" to see if the student will Pass or Fail

📂 Dataset
The project uses a dataset named exams.csv with the following columns:
gender
race/ethnicity (removed in preprocessing)
parental level of education
lunch
test preparation course
math score


reading score

writing score
