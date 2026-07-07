import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
df = pd.read_csv("exam_scores.csv")
df["pass"] = df["score"] >= 70

X = df[["hours_studied", "age"]]   
y = df["pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)
print("Actual:", y_test.values)
print("Predicted:", predictions)
