from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections
from sklearn.metrics import confusion_matrix
data = pd.read_csv("creditcard.csv")
d1=np.array(data['Amount'])
data['normAmount'] = StandardScaler().fit_transform(d1.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
print("data size",collections.Counter(y['Class']))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)
print("test data size",collections.Counter(y_test['Class']))
print("original training data size",collections.Counter(y_train['Class']))
y_train_arr=np.array(y_train['Class'])
X_train_arr=np.array(X_train)
A=[1,3,5,7,9]    
for i in A:
    nm = NearMiss(version=3,random_state=5,n_neighbors_ver3=7)
    #ratio after sampling Nmin/Mmaj n_neighbors=5 number of neighbour to be taken in consideration at a time
    X_train_sampled,y_train_sampled  = nm.fit_sample(X_train_arr, y_train_arr)
    print("sampled training data size",collections.Counter(y_train_sampled))
    
    #random forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    clf.fit(X_train_sampled,y_train_sampled)
    y_pred=clf.predict(X_test)
    cn_mat=confusion_matrix( y_pred,y_test['Class'])
    print(cn_mat)
    TP=cn_mat[1][1]
    TN=cn_mat[0][0]
    FN=cn_mat[0][1]
    FP=cn_mat[1][0]
    pre1=TP/(TP+FP)
    recall1=TP/(TP+FN)
    NPV1=TN/(TN+FN)
    speci1=TN/(TN+FP)
    print("Precision : ",pre1,"Recall : ",recall1,"Negative Predictive Rate(TN/(TN+FN)) : ",NPV1,"Specificity : ",speci1)
    F1=2*(pre1*recall1)/(pre1+recall1)
    F2=2*(NPV1*speci1)/(NPV1+speci1)
    print("F1 : ",F1,"F2 : ",F2)
    y_test_arr=np.array(y_test['Class'])
    print(classification_report(y_test_arr, y_pred))
