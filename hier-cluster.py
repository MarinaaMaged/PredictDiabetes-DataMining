import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


df = pd.read_csv(r"D:\MiningFinal\project\csv_files\cleaned_diabetes.csv")


X = df.iloc[:, :2].values


linked = linkage(X, "single")


plt.figure(figsize=(10, 7))
dendrogram(linked, orientation="top", distance_sort="descending", show_leaf_counts=True)

dendrogram(linked, truncate_mode="lastp", p=10)
plt.show()
