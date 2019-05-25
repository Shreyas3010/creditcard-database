from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
d1=np.array(data['Amount'])
data['normAmount'] = StandardScaler().fit_transform(d1.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
datasize=collections.Counter(y['Class'])
print("data size",collections.Counter(y['Class']))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)
testingdatasize=collections.Counter(y_test['Class'])
print("test data size",testingdatasize)
trainingdatasize=collections.Counter(y_train['Class'])
print("original training data size",trainingdatasize)
numofmaj=trainingdatasize[0]
numofmin=trainingdatasize[1]
row1=np.arange(648)
results= pd.DataFrame(data=None,index=row1,columns = ['ratio SMOTE','ratio NM02','Class','Datasize','Training Datasize','After sampling SMOTE','After sampling NM02','Precision','Recall','f1-score','oobscore','AUC','Testing Datasize'])
#0.6 is limit
A=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]    
B=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
a=0
print(a)
for i in A:
    rat=numofmin/(numofmaj*i)
    if rat <= 1:
        sm=SMOTE(sampling_strategy=rat,random_state=5)
        #ratio after sampling Nmin/Mmaj n_neighbors=5 number of neighbour to be taken in consideration at a time
        y_train_arr=np.array(y_train['Class'])
        X_train_arr=np.array(X_train)
        X_train_sampled1,y_train_sampled1  = sm.fit_sample(X_train_arr, y_train_arr)
        samplingdatasize1=collections.Counter(y_train_sampled1)
        for j in B:
            print("---------------")
            print("ratio RUS",i)
            print("sampling_strategy SMOTE",rat)
            results['ratio SMOTE'][a]=i
            print("ratio NM02",j)
            results['ratio NM02'][a]=j
            b=a
            a=a+1
            results['Class'][b]=0
            results['Class'][a]=1
            results['Datasize'][b]=datasize[0]
            results['Datasize'][a]=datasize[1]
            results['Training Datasize'][b]=trainingdatasize[0]
            results['Training Datasize'][a]=trainingdatasize[1]
            results['Testing Datasize'][b]=testingdatasize[0]
            results['Testing Datasize'][a]=testingdatasize[1]
            numofmaj2=samplingdatasize1[0]
            numofmin2=samplingdatasize1[1]
            rat2=numofmin2/(numofmaj2*j)
            print("sampling_strategy RUS",rat2)
            print("sampled training data size SMOTE ",samplingdatasize1)
            results['After sampling SMOTE'][b]=samplingdatasize1[0]
            results['After sampling SMOTE'][a]=samplingdatasize1[1]
            if rat2 <= 1:
                nm = NearMiss(sampling_strategy=rat,version=2,random_state=5)
                X_train_sampled,y_train_sampled  = nm.fit_sample(X_train_sampled1,y_train_sampled1)
                samplingdatasize=collections.Counter(y_train_sampled)
                print("sampled training data size RUS",samplingdatasize)
                results['After sampling NM02'][b]=samplingdatasize[0]
                results['After sampling NM02'][a]=samplingdatasize[1]
                
                
                #random forest
                clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0,oob_score=True)
                clf.fit(X_train_sampled,y_train_sampled)
                y_pred=clf.predict(X_test)
                y_test_arr=np.array(y_test['Class'])
                oobscore=clf.oob_score_
                print("oob score",oobscore)
                results['oobscore'][b]=round(oobscore,5)
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
                results['Precision'][b]=round(NPV1,5)
                results['Recall'][b]=round(speci1,5)
                results['Precision'][a]=round(pre1,5)
                results['Recall'][a]=round(recall1,5)
                print("Precision : ",pre1,"Recall : ",recall1,"Negative Predictive Rate(TN/(TN+FN)) : ",NPV1,"Specificity : ",speci1)
                F1=2*(pre1*recall1)/(pre1+recall1)
                F2=2*(NPV1*speci1)/(NPV1+speci1)
                results['f1-score'][b]=round(F2,5)
                results['f1-score'][a]=round(F1,5)
                print("F1 : ",F1,"F2 : ",F2)
                print(classification_report(y_test_arr, y_pred))
                fpr, tpr, _ = roc_curve(y_test,y_pred)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(10,10))
                lw = 2
                plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                results['AUC'][b]=roc_auc
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                i1=str(i)
                j1=str(j)
                plt.title('Receiver operating characteristic example SMOTE'+i1+'NM02'+j1)
                plt.legend(loc="lower right")
                plt.savefig('Graph/SMOTE'+i1+'NM02'+j1+'.png')
                plt.show()
            a=a+1
            print("a",a)


            
        

