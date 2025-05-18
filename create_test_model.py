import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Create a simple random forest classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Create some dummy training data (3 channel images, 224x224)
X_train = np.random.rand(100, 224 * 224 * 3)  # 100 sample images
y_train = np.random.choice(['apple', 'banana', 'orange'], size=100)  # Random labels

# Train the model
model.fit(X_train, y_train)

# Save the model
with open('model/fruit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Test model created and saved successfully!") 