import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:\MiningFinal\project\csv_files\cleaned_diabetes.csv")


# columns for clustering
diabetes = df[["Glucose", "BloodPressure", "BMI", "Age"]]


k_values = range(1, 11)

# I want to know the optimal number of clusters so I have used Elbow method
wcss = []
for k in k_values:
    kmedoids = KMedoids(n_clusters=k).fit(diabetes)
    wcss.append(kmedoids.inertia_)

# Plot the elbow curve
plt.plot(k_values, wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# KMediods
k = 4

kmedoids = KMedoids(n_clusters=k).fit(diabetes)

clusters = diabetes.iloc[kmedoids.medoid_indices_]

# Get cluster labels
labels = kmedoids.labels_

print("Labels:", labels, "\n")
print("Cluster Centers (Medoids):\n", clusters, "\n")

for j in range(k):
    for i in range(len(diabetes)):
        if kmedoids.labels_[i] == j:
            x = diabetes.iloc[i]
            print("Cluster", j, ":\n", diabetes.iloc[i])

columns = diabetes.columns
n_columns = len(columns)

fig, axes = plt.subplots(n_columns, n_columns, figsize=(15, 15))

cluster_scatters = {}

for i in range(n_columns):
    for j in range(n_columns):
        ax = axes[i, j]
        if i == j:
            ax.axis("off")
        else:
            for cluster_id in range(k):
                cluster_points = diabetes[labels == cluster_id]
                scatter = ax.scatter(
                    cluster_points[columns[j]],
                    cluster_points[columns[i]],
                    label=f"Cluster {cluster_id}",
                    alpha=0.5,
                )
                ax.scatter(
                    clusters.iloc[cluster_id, j],
                    clusters.iloc[cluster_id, i],
                    c="blue",
                    marker="x",
                    s=100,
                    label=f"Cluster {cluster_id} Center",
                )

                if cluster_id not in cluster_scatters:
                    cluster_scatters[cluster_id] = scatter

            ax.set_xlabel(columns[j])
            ax.set_ylabel(columns[i])

handles, labels = [], []
for cluster_id, scatter in cluster_scatters.items():
    handles.append(scatter)
    labels.append(f"Cluster {cluster_id}")

fig.legend(handles, labels, loc="center right")

plt.tight_layout()
plt.show()
