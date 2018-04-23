import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

df=pd.read_csv('data.csv')

Y=df['emotion']
X=df.drop(['emotion'],axis=1)

normalized_X = preprocessing.normalize(X)

X_train,X_test,Y_train,Y_test=train_test_split(normalized_X,Y,test_size=0.25)



clf = SVC(kernel='linear')
#clf.fit(X_train,Y_train)
scores = cross_val_score(clf, normalized_X, Y,cv=2)
print (scores)