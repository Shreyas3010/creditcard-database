import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import collections
from sklearn.model_selection import GridSearchCV

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
#ratio after sampling Nmin/Mmaj n_neighbors=5 number of neighbour to be taken in consideration at a time
#print("original data size class 1: ",len(y.loc[y['Class'] == 1]))
#print("original data size class 0: ",len(y.loc[y['Class'] == 0]))
#random forest
weights=np.linspace(0.05,0.95,20)
#clf = RandomForestClassifier(n_estimators=100,random_state=0,oob_score=True,class_weight='balanced')
clf = RandomForestClassifier(n_estimators=100,random_state=0,oob_score=True)
parameters={'class_weight':[{0:x,1:1.0-x} for x in weights]}
gsc=GridSearchCV(clf,parameters,scoring='f1',cv=3)
grid_res=gsc.fit(X_train,y_train.values.ravel())
grid_best_clf=grid_res.best_estimator_
print("best grid clf : ",grid_best_clf)
grid_best_clf.fit(X_train,y_train.values.ravel())
oobscore=grid_best_clf.oob_score_
X_test_arr=np.array(X_test)
y_pred=grid_best_clf.predict(X_test_arr)
print("oob score",oobscore)
#feature_imp=grid_best_clf.feature_importances_
#print("feature importances",feature_imp)
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
