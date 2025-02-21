import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (Replace with your GitHub Raw CSV link)
url = "https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/YOUR-DATASET.csv"
data = pd.read_csv(url)

# Handle missing values
data.dropna(inplace=True)

# Split features and target variable
X = data.drop(columns=['Risk'])  # Features
y = data['Risk']  # Target

# Convert categorical variables (if any)
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
