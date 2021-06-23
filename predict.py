# Importing required libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import sklearn.metrics as metrics


# Loading datasets
iris = load_iris()

# Convert to pandas dataframe
iris_data = pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris_data.head()

# printing categories (setosa, versicolor, virginica)
print(iris.target_names)
# print flower features
print(iris.feature_names)

# setting independent (X) and dependent (Y) variables
X = iris_data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
Y = iris_data['species']  # Labels


# printing feature data
print(X[0:5])
# printing dependent variable values (0 = setosa, 1 = versicolor, 3 = virginica)
print(Y)

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

# defining random forest classifier
clfr = RandomForestClassifier(random_state = 100)
clfr.fit(X_train, y_train)

# making prediction
Y_pred = clfr.predict(X_test)

# checking model accuracy
print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))
cm = np.array(confusion_matrix(y_test, Y_pred))
print(cm)

# making predictions on new data
species_id = clfr.predict([[5.1, 3.5, 1.4, 0.2]])
iris.target_names[species_id]
print(iris.target_names[species_id])

# determining feature importance (e.g. model participation)
feature_imp = pd.Series(clfr.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot to visualize feature participation in model
sns.barplot(x=feature_imp, y=feature_imp.index)

# use '%matplotlib inline' to plot inline in jupyter notebooks
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
