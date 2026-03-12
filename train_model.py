import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Toy dataset
data = {
    "experience":[1,2,3,4,5,6],
    "salary":[30000,40000,50000,60000,70000,80000]
}

df = pd.DataFrame(data)

X = df[["experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X,y)

# Save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved successfully")