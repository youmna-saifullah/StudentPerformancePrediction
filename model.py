import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Load dataset
df = pd.read_csv("exams.csv")

# Drop race/ethnicity
df = df.drop(columns=['race/ethnicity'])

# Create Pass/Fail target based on average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'parental level of education', 'lunch', 'test preparation course', 'result']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop(columns=['result'])
y = df['result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")


# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=label_encoders['result'].classes_.tolist(),
    filled=True
)
plt.show()


# ====== Tkinter UI ======
def predict_result():
    try:
        # Get user input
        gender = gender_var.get()
        parent_edu = parent_edu_var.get()
        lunch = lunch_var.get()
        prep_course = prep_var.get()
        math = float(math_score_var.get())
        reading = float(reading_score_var.get())
        writing = float(writing_score_var.get())

        avg_score = (math + reading + writing) / 3

        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            'gender': label_encoders['gender'].transform([gender])[0],
            'parental level of education': label_encoders['parental level of education'].transform([parent_edu])[0],
            'lunch': label_encoders['lunch'].transform([lunch])[0],
            'test preparation course': label_encoders['test preparation course'].transform([prep_course])[0],
            'math score': math,
            'reading score': reading,
            'writing score': writing,
            'average_score': avg_score
        }])

        # Predict
        prediction = model.predict(input_df)[0]
        result_label = label_encoders['result'].inverse_transform([prediction])[0]
        messagebox.showinfo("Prediction", f"The student will: {result_label}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create Tkinter window
root = tk.Tk()
root.title("Student Performance Prediction")

# Dropdown options
genders = label_encoders['gender'].classes_
parent_edu_options = label_encoders['parental level of education'].classes_
lunch_options = label_encoders['lunch'].classes_
prep_options = label_encoders['test preparation course'].classes_

# Variables
gender_var = tk.StringVar(value=genders[0])
parent_edu_var = tk.StringVar(value=parent_edu_options[0])
lunch_var = tk.StringVar(value=lunch_options[0])
prep_var = tk.StringVar(value=prep_options[0])
math_score_var = tk.StringVar()
reading_score_var = tk.StringVar()
writing_score_var = tk.StringVar()

# UI layout
tk.Label(root, text="Gender").grid(row=0, column=0)
tk.OptionMenu(root, gender_var, *genders).grid(row=0, column=1)

tk.Label(root, text="Parental Education").grid(row=1, column=0)
tk.OptionMenu(root, parent_edu_var, *parent_edu_options).grid(row=1, column=1)

tk.Label(root, text="Lunch Type").grid(row=2, column=0)
tk.OptionMenu(root, lunch_var, *lunch_options).grid(row=2, column=1)

tk.Label(root, text="Test Preparation").grid(row=3, column=0)
tk.OptionMenu(root, prep_var, *prep_options).grid(row=3, column=1)

tk.Label(root, text="Math Score").grid(row=4, column=0)
tk.Entry(root, textvariable=math_score_var).grid(row=4, column=1)

tk.Label(root, text="Reading Score").grid(row=5, column=0)
tk.Entry(root, textvariable=reading_score_var).grid(row=5, column=1)

tk.Label(root, text="Writing Score").grid(row=6, column=0)
tk.Entry(root, textvariable=writing_score_var).grid(row=6, column=1)

tk.Button(root, text="Predict", command=predict_result).grid(row=7, column=0, columnspan=2)

root.mainloop()
