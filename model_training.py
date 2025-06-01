# model_training.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Generate sample dataset
df = pd.DataFrame({
    'feature1': range(1, 101),
    'feature2': [x * 2 for x in range(1, 101)],
})
df['target'] = df['feature1'] * 3 + df['feature2'] * 2 + 5

# Train model
X = df[['feature1', 'feature2']]
y = df['target']

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
