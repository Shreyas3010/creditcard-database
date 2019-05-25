
from imblearn.under_sampling import CondensedNearestNeighbour
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

#samppling

cnn = CondensedNearestNeighbour(random_state=1)
X_sampled,y_sampled  = cnn.fit_sample(X, y.values.ravel())
print("sampled data size",collections.Counter(y_sampled))
X_train, X_test, y_train, y_test = train_test_split(X_sampled,y_sampled,test_size = 0.3, random_state = 0)
X_train_sampled_df = pd.DataFrame(X_train)
y_train_sampled_df = pd.DataFrame(y_train)
X_test_sampled_df = pd.DataFrame(X_test)
y_test_sampled_df = pd.DataFrame(y_test)

#random forest
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_train_sampled_df,y_train_sampled_df.values.ravel())
y_pred=clf.predict(X_test_sampled_df)
print("predicted")
print(clf.predict(X_test_sampled_df))
print("actual")
print(y_test_sampled_df)
print(classification_report(y_test_sampled_df, y_pred))
