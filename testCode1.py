import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris
import seaborn as sns
import statsmodels.api as sm

# Load the data
results_df = pd.read_csv("Marks-2018.csv")
results_scaled_df = pd.read_csv("Marks-2018-Scaled.csv")

# Define the mode function
def mmode(v):
    uniqv = np.unique(v)
    return uniqv[np.argmax(np.bincount(np.where(v == uniqv[:, None])[0]))]

# Section 1: Data Preparation (Load and Clean)
# Change column names to date formatted string in Sales
results_df.columns = [col.replace("X", "").replace(".", "/") for col in results_df.columns]

# Convert Rank variables to factors
results_df['Rank'] = results_df['Rank'].astype('category')
results_scaled_df['Rank'] = results_scaled_df['Rank'].astype('category')

# Take a subset of data which are marks and Rank for analysis
marks_df = results_df.iloc[:, 4:11]
marks_scaled_df = results_scaled_df.iloc[:, 4:11]

# Collect the required features and handle missing values
marks_df.dropna(inplace=True)
marks_scaled_df.dropna(inplace=True)

# Section 2: Exploratory Analysis
# Normalize data
scaler = StandardScaler()
marks_scaled = scaler.fit_transform(marks_df)

# Skewness and Kurtosis
skewness = marks_df['T.Marks'].skew()
kurtosis = marks_df['T.Marks'].kurt()

# Compute standard deviation
std_dev = marks_df['T.Marks'].std()

# Assume that 2 std should cover all data
LT = marks_df['T.Marks'].mean() - 2 * std_dev
UT = marks_df['T.Marks'].mean() + 2 * std_dev

# Filter data based on bounds
outliers_lower = marks_df[marks_df['T.Marks'] < LT]
outliers_upper = marks_df[marks_df['T.Marks'] > UT]

# Correlation matrix
corr_marks = marks_df.corr()
corr_scaled_marks = marks_scaled_df.corr()

# Plot histogram of the number of students per school
school_volume = results_df['School'].value_counts()
plt.hist(school_volume, bins=20, color='skyblue')
plt.xlabel("Number of Students per School")
plt.ylabel("Number of Schools")
plt.title("Distribution of Student Count")
plt.show()

# Check multi-collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = marks_df.iloc[:, :-1]  # Exclude the dependent variable
vif = pd.DataFrame()
vif["Features"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

# Section 3: Clustering
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(marks_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 16), wcss)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(marks_scaled)
marks_df['Cluster'] = cluster_labels

# Summary statistics by cluster
cluster_summary = marks_df.groupby('Cluster')['T.Marks'].agg(['mean', 'median'])
print(cluster_summary)

# Fit Linear Regression on the cluster dataframes
X = sm.add_constant(marks_df.drop(['T.Marks', 'Cluster'], axis=1))
y = marks_df['T.Marks']
model = sm.OLS(y, X).fit()
print(model.summary())
