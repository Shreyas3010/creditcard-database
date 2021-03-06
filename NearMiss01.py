from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections

data = pd.read_csv("creditcard.csv")
d1=np.array(data['Amount'])
data['normAmount'] = StandardScaler().fit_transform(d1.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
print("data size",collections.Counter(y['Class']))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)
print("test data size",collections.Counter(y_test['Class']))
traindatasize=collections.Counter(y_train['Class'])
print("original training data size",traindatasize)
numofmaj=traindatasize[0]
numofmin=traindatasize[1]
y_train_arr=np.array(y_train['Class'])
X_train_arr=np.array(X_train)
A=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]    
for i in A:
    rat=numofmin/(numofmaj*i)
    print("---------------")
    print("ratio",i)
    print("sampling_strategy",rat)
    nm = NearMiss(sampling_strategy=rat,version=1,random_state=5,n_neighbors=3)
    #ratio after sampling Nmin/Mmaj n_neighbors=5 number of neighbour to be taken in consideration at a time
    X_train_sampled,y_train_sampled  = nm.fit_sample(X_train_arr, y_train_arr)
    #print("original data size class 1: ",len(y.loc[y['Class'] == 1]))
    #print("original data size class 0: ",len(y.loc[y['Class'] == 0]))
    print("sampled training data size",collections.Counter(y_train_sampled))
    
    #random forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    clf.fit(X_train_sampled,y_train_sampled)
    X_test_arr=np.array(X_test)
    y_pred=clf.predict(X_test_arr)
    print("predicted")
    print(y_pred)
    print("actual")
    y_test_arr=np.array(y_test['Class'])
    print(y_test_arr)
    print(classification_report(y_test_arr, y_pred))
