import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv("data/Iris.csv")

# Drop ID
df = df.drop("Id", axis=1)

# Encode target
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")