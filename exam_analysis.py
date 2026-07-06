import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("exam_scores.csv")

X = df[["hours_studied"]]   
y = df["score"]
model = LinearRegression()
model.fit(X, y)

predicted = model.predict([[6]])
print("Predicted score for 6 hours studied:", predicted)