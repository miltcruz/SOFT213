# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("dataset.csv")

# Separate features (X) and target (y)
X = data[["marketing_budget", "social_media_followers"]]
y = data["ticket_sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print predictions
print("Predictions:")
print(predictions.round().astype(int))

# Print actual values
print("\nActual values:")
print(y_test.values)