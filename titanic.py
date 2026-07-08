import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
df = pd.read_csv("titanic.csv")
df = df.drop(columns=["Cabin","Name", "Ticket", "PassengerId"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"])
X = df[["Age","Embarked_C", "Embarked_Q", "Embarked_S", "Sex", "Pclass", "SibSp", "Parch", "Fare"]]   
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)
print("Actual:", y_test.values)
print("Predicted:", predictions)