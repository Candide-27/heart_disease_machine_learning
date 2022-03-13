import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#import and change column names
df = pd.read_csv('processed.cleveland.data',
                header=None)
column_dict = dict(
    age='age',
    sex='sex',
    cp='chest pain',
    restbp='resting blood pressure',
    chol='serum cholesterol',
    fbs='fasting blood sugar',
    restecg='resting electrocardiographic',
    thalach='maximum heart rate achieved',
    exang='exercise induced angina',
    oldpeak='ST depression induced by exercise relative to rest',
    slope='the slope of the peak exercise ST segment',
    ca='number of major vessels colored by fluoroscopy',
    thal='short of thalium heart scan',
    hd='diagnosis of heart disease - the predicted attribute'
)
df.columns = [k for k in column_dict.keys()]
#print(df.head())

#drop the '?'
#print(df[['ca,'thal]].unique())
df = df[ (df['ca'] != '?') & (df['thal'] != '?') ]


#split data into y and x
X = df.iloc[:,:13].copy()
y = df.iloc[:,13].copy()
#change categorical variable to dummies
X = pd.get_dummies(X, columns=['cp','restecg', 'slope','thal'])
#change heart disease to a binary
y[y>0] = 1

#Split Data to Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Create a classification tree object
clf_tree = DecisionTreeClassifier(criterion='gini',random_state=0)
clf_tree = clf_tree.fit(X_train, y_train) #fit on the training sample only
fig, ax = plt.subplots(1,1, figsize=(12,6))
plot_tree(clf_tree, filled=True, rounded=True, class_names=['No HD', 'Yes HD'])
plt.show()

fig.savefig('DecisionTree.jpg')
