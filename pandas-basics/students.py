import pandas as pd
df = pd.read_csv("students.csv")
print(df)
print(df.head())
print(df["age"])
print(df["age"].mean())
print(df.describe())
print(df["course"].value_counts())
aiml_stud = df[df["course"] == "AIML"]
print(aiml_stud)
print(df[df["course"]=="AIML"])
print(df["course"] == "AIML")
print(df[df["age"] > 21])