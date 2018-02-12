import pandas as pd
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys

input_file = '/home/lia/Documents/the_project/dataset/iris_imb.csv'
df = pd.read_csv(input_file)

print(len(df))

X = df[['sepallength','sepalwidth','petallength','petalwidth']]
y = df[['class']]

smote = SMOTE(kind = "regular")
X_sm, y_sm = smote.fit_sample(X, y.values.ravel())

X_sm = pd.DataFrame(X_sm, index=None,columns=['sepallength','sepalwidth','petallength','petalwidth'])
y_sm = pd.DataFrame(y_sm, index=None, columns=['class'])

print(X)
print(len(X_sm))
print(X_sm)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_sm)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y_sm], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# Show the plot
plt.show()
