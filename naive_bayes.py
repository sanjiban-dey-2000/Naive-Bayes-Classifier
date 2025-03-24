# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""**IMPORTING DATASET**"""

dataset=pd.read_csv('Social_Network_Ads.csv')
dataset.head()

x=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]

"""**IMPORTING DEPENDENCIES**"""

x.head()

y.head()

"""**SPLITTING DATASET INTO TRAIN AND TEST DATA**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

x_train.shape

x_test.shape

"""**FEATURE SCALING**"""

from sklearn.preprocessing import StandardScaler
ss_x=StandardScaler()
x_train=ss_x.fit_transform(x_train)
x_test=ss_x.transform(x_test)

x_train

x_test

"""**FITTING NAIVE BAYES TO TRAINING SET**"""

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

"""**PREDICTING TEST RESULT**"""

y_pred=classifier.predict(x_test)
y_pred

"""**CONFUSION MATRIX**"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

"""**PREDICTING THE ACCURACY**"""

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

"""**VISUALIZING TRANING SET**"""

from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train

X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

cmap = ListedColormap(['#800080', '#008000'])

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap=cmap)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = cmap(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

plt.show()

"""**VISUALIZING TEST DATA**"""

from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

cmap = ListedColormap(['purple', 'green'])

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap=cmap)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = [cmap(i)], label = j)
plt.title('Naive Bayes (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
