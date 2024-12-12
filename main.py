import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
def load_data():
    # Load the complete dataset from a CSV file or other sources
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    data.columns = data.columns.str.lower()  # Normalize column names to lowercase
    print("Columns in dataset:", data.columns)
    return data

# Preprocess the data
def preprocess_data(data):
    label_encoder = LabelEncoder()

    # Handle categorical variables dynamically
    if "gender" in data.columns:
        data["gender"] = label_encoder.fit_transform(data["gender"])
    else:
        print("Warning: 'gender' column not found in dataset!")

    if "smoking_history" in data.columns:
        data["smoking_history"] = label_encoder.fit_transform(data["smoking_history"])
    else:
        print("Warning: 'smoking_history' column not found in dataset!")

    # Ensure target column exists
    if "diabetes" not in data.columns:
        raise ValueError("The dataset must contain a 'diabetes' column for the target variable.")

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model using Random Forest and GridSearchCV
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Diabetic", "Diabetic"], yticklabels=["Non-Diabetic", "Diabetic"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("confusion_matrix.png")
    print("Confusion Matrix saved as 'confusion_matrix.png'")

    # Feature Importance Visualization
    feature_importances = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
    plt.title("Feature Importances")
    plt.savefig("feature_importances.png")
    print("Feature Importances saved as 'feature_importances.png'")

# Manually enter data from CLI
def get_user_input():
    print("Enter the following details:")
    gender = input("Gender (0 for Female, 1 for Male): ")
    age = input("Age: ")
    hypertension = input("Hypertension (0 for No, 1 for Yes): ")
    heart_disease = input("Heart Disease (0 for No, 1 for Yes): ")
    smoking_history = input("Smoking History (0 for Never, 1 for Former, 2 for Current): ")
    bmi = input("BMI: ")
    hba1c_level = input("HbA1c Level: ")
    blood_glucose_level = input("Blood Glucose Level: ")

    return [
        float(gender), float(age), int(hypertension), int(heart_disease), int(smoking_history),
        float(bmi), float(hba1c_level), float(blood_glucose_level)
    ]

# Visualize user's data point on charts
def visualize_user_data(model, X_test, user_data):
    # Add user data to the dataset
    user_array = np.array(user_data).reshape(1, -1)
    user_proba = model.predict_proba(user_array)[0][1]  # Probability of being diabetic

    # Plot user's data against existing blood glucose levels
    plt.figure(figsize=(8, 6))
    sns.histplot(X_test['blood_glucose_level'], kde=True, color="blue", label="Population")
    plt.axvline(user_data[-1], color="red", linestyle="--", label="User's Blood Glucose Level")
    plt.title("User's Blood Glucose Level Compared to Population")
    plt.legend()
    plt.xlabel("Blood Glucose Level")
    plt.ylabel("Density")
    plt.savefig("user_blood_glucose_comparison.png")
    print("User's Blood Glucose Level comparison saved as 'user_blood_glucose_comparison.png'")

    # Probability of being diabetic
    plt.figure(figsize=(6, 4))
    plt.bar(["Non-Diabetic", "Diabetic"], model.predict_proba(user_array)[0], color=['blue', 'red'])
    plt.title("Prediction Probability for User Data")
    plt.ylabel("Probability")
    plt.savefig("user_prediction_probability.png")
    print("Prediction Probability saved as 'user_prediction_probability.png'")

# Make a prediction
def predict(model, input_data):
    prediction = model.predict([input_data])
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Manual input and prediction
    user_data = get_user_input()
    result = predict(model, user_data)
    print("Prediction based on entered data:", result)

    # Visualize user's data point
    visualize_user_data(model, X_test, user_data)
