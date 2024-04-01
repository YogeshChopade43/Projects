import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load your dataset
df = pd.read_csv("store.csv")

print('\n')
print(df.info())
print('\n')
print(df.describe())


# Check for missing values in the dataset
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Fill missing values with the mean of each column
df = df.fillna(df.mean())

# Checking for duplicates
print("Duplicate values:")
print(df.duplicated())

# Dropping Duplicate rows
df.drop_duplicates(inplace=True)

# Extract relevant features
X = df[['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']].values

# Perform divisive clustering
divisive_linkage_matrix = linkage(X, method='complete')  # Divisive clustering linkage
divisive_labels = fcluster(divisive_linkage_matrix, 4, criterion='maxclust')

# Perform agglomerative clustering
agglomerative_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
agglomerative_labels = agglomerative_model.fit_predict(X)

# Plot dendrograms for divisive and agglomerative clustering side by side
plt.figure(figsize=(16, 8))

# Dendrogram for divisive clustering
plt.subplot(1, 2, 1)
dendrogram(divisive_linkage_matrix, truncate_mode='level', p=3, labels=df.index, orientation='top')
plt.title('Dendrogram for Divisive Clustering')

# Dendrogram for agglomerative clustering
plt.subplot(1, 2, 2)
agglomerative_linkage_matrix = linkage(X, method='ward')  # Agglomerative clustering linkage
dendrogram(agglomerative_linkage_matrix, truncate_mode='level', p=3, labels=df.index, orientation='top')
plt.title('Dendrogram for Agglomerative Clustering')

plt.tight_layout()
plt.show()

# Plot scatter plots for both divisive and agglomerative clustering on the same window
plt.figure(figsize=(12, 6))

# Scatter plot for divisive clustering
plt.subplot(1, 2, 1)
plt.scatter(df['Store_Area'], df['Store_Sales'], c=divisive_labels, cmap='Set1', s=50, edgecolors='k', label='Data Points')
plt.title(f'Divisive Clustering\nSilhouette: {silhouette_score(X, divisive_labels):.2f}, DB Index: {davies_bouldin_score(X, divisive_labels):.2f}, CH Index: {calinski_harabasz_score(X, divisive_labels):.2f}')
plt.xlabel('Store Area')
plt.ylabel('Store Sales')
plt.legend()

# Scatter plot for agglomerative clustering
plt.subplot(1, 2, 2)
plt.scatter(df['Store_Area'], df['Store_Sales'], c=agglomerative_labels, cmap='Set1', s=50, edgecolors='k', label='Data Points')
plt.title(f'Agglomerative Clustering\nSilhouette: {silhouette_score(X, agglomerative_labels):.2f}, DB Index: {davies_bouldin_score(X, agglomerative_labels):.2f}, CH Index: {calinski_harabasz_score(X, agglomerative_labels):.2f}')
plt.xlabel('Store Area')
plt.ylabel('Store Sales')
plt.legend()

plt.tight_layout()
plt.show()

# Box plots for relevant columns using seaborn
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x=df['Store_Area'], color='lightcoral')
plt.title('Box Plot for Store Area')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['Items_Available'], color='lightgreen')
plt.title('Box Plot for Items Available')

plt.subplot(2, 2, 3)
sns.boxplot(x=df['Daily_Customer_Count'], color='skyblue')
plt.title('Box Plot for Daily Customer Count')

plt.subplot(2, 2, 4)
sns.boxplot(x=df['Store_Sales'], color='gold')
plt.title('Box Plot for Store Sales')

plt.tight_layout()
plt.show()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pair plot for numeric features using seaborn
sns.set(style="ticks")
pair_plot = sns.pairplot(df[['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']])
pair_plot.fig.suptitle('Pair Plot for Numeric Features', y=1.02)
plt.show()  # Understanding pairwise relationships and patterns

# Histograms for selected features in a 2x2 format
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Histogram for Store Area
axs[0, 0].hist(df['Store_Area'], bins=20, color='lightcoral', edgecolor='black')
axs[0, 0].set_title('Histogram for Store Area')
axs[0, 0].set_xlabel('Store Area')
axs[0, 0].set_ylabel('Frequency')

# Histogram for Store Sales
axs[0, 1].hist(df['Store_Sales'], bins=20, color='lightgreen', edgecolor='black')
axs[0, 1].set_title('Histogram for Store Sales')
axs[0, 1].set_xlabel('Store Sales')
axs[0, 1].set_ylabel('Frequency')

# Histogram for Daily Customer Count
axs[1, 0].hist(df['Daily_Customer_Count'], bins=20, color='skyblue', edgecolor='black')
axs[1, 0].set_title('Histogram for Daily Customer Count')
axs[1, 0].set_xlabel('Daily Customer Count')
axs[1, 0].set_ylabel('Frequency')

# Histogram for Items Available
axs[1, 1].hist(df['Items_Available'], bins=10, color='gold', edgecolor='black')
axs[1, 1].set_title('Histogram for Items Available')
axs[1, 1].set_xlabel('Items Available')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()

# Inference: Histograms showing the distribution of selected features.
plt.annotate('Histograms showing\nthe distribution of\nselected features', xy=(0.5, -0.15), ha='center', va='center',
             xycoords='axes fraction', textcoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='none'))

plt.show()

# Print Performance Metrics for comparison
print("\nDivisive Clustering Metrics:")
print(f"Silhouette Score: {silhouette_score(X, divisive_labels):.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X, divisive_labels):.2f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, divisive_labels):.2f}")


print("\nAgglomerative Clustering Metrics:")
print(f"Silhouette Score: {silhouette_score(X, agglomerative_labels):.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X, agglomerative_labels):.2f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, agglomerative_labels):.2f}")

# Calculate correlation matrix
correlation_matrix = df[['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']].corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

