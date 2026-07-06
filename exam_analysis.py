import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("exam_scores.csv")

X = df[["hours_studied"]]   
y = df["score"]
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X_test)
print("Actual scores:", y_test.values)
print("Predicted scores:", predictions)

error = mean_absolute_error(y_test, predictions)
print("Average error:", error)
