import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Toy dataset
data = {
    "experience":[1,2,3,4,5,6],
    "salary":[30000,40000,50000,60000,70000,80000]
}

data = pd.DataFrame(data)

X = data[["experience"]]
y = data["salary"]
#train model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(X,y)

# Save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved successfully")
