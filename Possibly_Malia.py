import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
from scipy.cluster.hierarchy import \
     (dendrogram,
      cut_tree)

df = pd.read_csv('/Users/maliakema/Desktop/BS data/Project/Master_num.csv')

ft=['Gender','Age','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','0be1dad','N']
X= df[ft]
y= df['BMI']
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Variance explained by 2 principal components')
pca = PCA(n_components=2)
Xpca = pca.fit_transform(X_train)
classifier_pca = LinearRegression()
classifier_pca.fit(Xpca, y_train)
X2 = pca.transform(X_test)
accuracy = classifier_pca.score(X2, y_test)
y_pred = classifier_pca.predict(X2)
print("Accuracy of 2 PC:", accuracy)

print('Variance explained by 3 principal components')
pca = PCA(n_components=3)
Xpca = pca.fit_transform(X_train)
classifier_pca = LinearRegression()
classifier_pca.fit(Xpca, y_train)
X2 = pca.transform(X_test)
accuracy = classifier_pca.score(X2, y_test)
y_pred = classifier_pca.predict(X2)
print("Accuracy of 3 PC:", accuracy)