
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
print("original training data size",collections.Counter(y_train['Class']))

#random forest
clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0,oob_score=True)

clf.fit(X_train,y_train.values.ravel())
score1=clf.score(X_train,y_train)
oobscore=clf.oob_score_
X_test_arr=np.array(X_test)
y_pred=clf.predict(X_test_arr)
#accscore=accuracy_score(X_test_arr,y_pred)
print("accuracy of classifier(score)",score1)
print("oob score",oobscore)
#print("accuracy (testing score) score",accscore)
feature_imp=clf.feature_importances_
print("feature importances",feature_imp)
y_act_count=collections.Counter(y_test['Class'])
y_pred_count=collections.Counter(y_pred)
print("predicted",y_pred_count)
print("actual",y_act_count)
y_test_arr=np.array(y_test['Class'])
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
print(classification_report(y_test['Class'], y_pred))
