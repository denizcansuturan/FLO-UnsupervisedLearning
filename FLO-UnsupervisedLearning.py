"""
Customer Segmentation with Unsupervised Learning

Business Problem
With Unsupervised Learning methods (Kmeans, Hierarchical Clustering), customers are desired to be clustered and their behaviors observed.

Data Story

The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases
in 2020-2021 via OmniChannel (both online and offline).
20,000 observations, 13 variables

master_id: Unique customer number
order_channel: Which channel was used for shopping on the platform (Android, iOS, Desktop, Mobile, Offline)
last_order_channel: The channel where the last purchase was made
first_order_date: The date of the customer's first purchase
last_order_date: The date of the customer's last purchase
last_order_date_online: The date of the customer's last online purchase
last_order_date_offline: The date of the customer's last offline purchase
order_num_total_ever_online: Total number of purchases made by the customer on the online platform
order_num_total_ever_offline: Total number of purchases made by the customer offline
customer_value_total_ever_offline: Total amount paid by the customer in offline purchases
customer_value_total_ever_online: Total amount paid by the customer in online purchases
interested_in_categories_12: List of categories in which the customer shopped in the last 12 months
store_type: Represents 3 different companies. If a person bought from A company and also from B, it is written as A,B.

TASKS

TASK 1: Data Preparation
1. Read the flo_data_20K.csv file.
"""
import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv("machine_learning/PART3/flo_data_20k.csv")
df = df_.copy()
df.head()
df.info()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


"""
2. Selecting the variables that will be used while segmenting customers. 
Creating new variables such as Tenure (customer age) and Recency (how many days ago the last purchase was made).
"""

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline","customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]
model_df.head()

"""
TASK 2: Customer Segmentation with K-MEANS
1. Standardize the variables.
"""
# This is a function that we use to check the skewness status of all variables;
# Skewness is checked to determine whether the distribution of the data follows a normal distribution.


def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return


"""plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)"""

# For those that do not follow a normal distribution;
# Log transformation can be applied to achieve normal distribution.

model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency'] = np.log1p(model_df['recency'])
model_df['tenure'] = np.log1p(model_df['tenure'])
model_df.head()

# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
model_df.head()
"""
We standardize the variables in the data set for several reasons, 
including the need to cluster customers into 12 variables.

-To prevent variables from dominating each other and to ensure that each variable is given equal weight in the analysis.
-When using gradient descent based optimization methods, standardization can positively contribute to optimization time.
-In distance-based methods such as similarity/dissimilarity, variable standardization is necessary to prevent variables 
from dominating each other.

"""

"""
2. Determine the optimum number of clusters.
"""

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)
# 7

"""
3. Create your model and segment your customers.
"""

k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments = k_means.labels_

final_df = df[["master_id","order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online", "recency", "tenure"]]
final_df["segment"] = segments
final_df.head()

"""
4. Statistically analyze each segment.
"""
final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})

"""
TASK 3: Customer Segmentation with Hierarchical Clustering
1. Using the standardized dataframe in Task 2, determining the optimum number of clusters.
"""

"""TWO METHODS TO BE USED:
Complete Linkage: Computes the farthest distance between two clusters.
Average Linkage: Computes the average distance between two clusters."""

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)
#5

"""
2. Create your model and segment your customers.
"""
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id", "order_num_total_ever_online","order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online", "recency", "tenure"]]
final_df["segment"] = segments
final_df.head()

"""
3. Statistically analyze each segment.
"""

final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})


