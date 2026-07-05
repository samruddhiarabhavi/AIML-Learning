import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("exam_scores.csv")

subject_avg = df.groupby("subject")["score"].mean()
sns.barplot(x=subject_avg.index, y=subject_avg.values)
plt.title("average score per subject")
plt.xlabel("subject")
plt.ylabel("Average score")
plt.show()