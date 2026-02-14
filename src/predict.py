import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

print("Enter flower measurements:")

try:
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))
except ValueError:
    print("Invalid input. Please enter numeric values only.")
    exit()

# Create DataFrame with correct column names
sample = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
)

# Scale
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("\nPredicted Species:", class_names[prediction[0]])