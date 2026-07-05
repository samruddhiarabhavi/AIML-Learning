import pandas as pd
df = pd.read_csv("exam_scores.csv")
print("Average score overall:", df["score"].mean())
print("Average score in Math:", df[df["subject"] == "Math"]["score"].mean())
print("Students per city:\n", df["city"].value_counts())
print("Average score per subject:\n", df.groupby("subject")["score"].mean())
print("Students scoring above 80:", len(df[df["score"] > 80]))