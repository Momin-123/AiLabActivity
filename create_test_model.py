import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

model = RandomForestClassifier(n_estimators=10, random_state=42)

X_train = np.random.rand(100, 224 * 224 * 3)  
y_train = np.random.choice(['apple', 'banana', 'orange'], size=100) 

model.fit(X_train, y_train)

with open('model/fruit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Test model created and saved successfully!") 