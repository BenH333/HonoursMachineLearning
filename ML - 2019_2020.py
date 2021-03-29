# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:46:19 2021

@author: Ben
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import Counter
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import math
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV

students_df = pd.read_csv('with_grades_df.csv')
students_df.rename( columns={'Unnamed: 0':'anonymous_id'}, inplace=True )

course_views = students_df['Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence'];

gradeDictionary = {5:'A',4:'B',3:'C',2:'D',1:'E',0:'NS'}

#%%###### TARGET LABEL ENCODER
l1 = LabelEncoder()

#fit will transform labels into a numeric value
l1.fit(students_df['OVERALL_GRADE'])

#transform the existing student_df dataset
students_df.OVERALL_GRADE = l1.transform(students_df.OVERALL_GRADE);
######

## intuitive transform
## reverse the label encode values
## previously 0=A, transform 0=NS, 1=E etc.
for idx, val in enumerate(students_df['OVERALL_GRADE']):
    if(val == 5):
        #print("grade is NS")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",5,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 0
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 4):
        #print("grade is E")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",4,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 1
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 3):
        #print("grade is D")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",3,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 2
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 2):
        #print("grade is C")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",2,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 3
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 1):
        #print("grade is B")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",1,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 4
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 0):
        #print("grade is A")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",0,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 5
        #print("new label:", students_df.loc[idx]['OVERALL_GRADE'],"\n")

#%%###### Course Views Logistic Regression

grades = students_df['OVERALL_GRADE']
course_grades = pd.concat([course_views,grades],axis='columns')

course_corr, _ = pearsonr(course_views, grades)
#print("Pearson course views linear relation with grade", course_corr)
course_grades = course_grades.rename(columns={'Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence':'course_views'})

#stdcon = sns.regplot(x='course_views',y='OVERALL_GRADE', data=course_grades)

#%% Course Views Logistic Regression & Outlier removal
def test_course_views(students_df):
    course_views = students_df['Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence'];
    grades = students_df['OVERALL_GRADE']
    course_grades = pd.concat([course_views,grades],axis='columns')
    
    course_grades = course_grades.rename(columns={'Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence':'course_views'})
    course_views = course_views.rename('course_views')
    #stdcon = sns.regplot(x='course_views',y='OVERALL_GRADE', data=course_grades)
    course_corr, _ = pearsonr(course_views, grades)
    print("Pearson course views linear relation with grade before", course_corr)
    
    #standard_dev = course_grades.std(axis = 0, skipna = True)
    #mean = course_grades.mean(axis = 0, skipna = True)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() 
    scaled = scaler.fit_transform(course_grades)
    
    ##DBSCAN detects outliers using a clustering method
    from sklearn.cluster import DBSCAN
    outlier_detection = DBSCAN(eps = 0.5, metric="euclidean", min_samples = 5, n_jobs = -1)
    clusters = outlier_detection.fit_predict(scaled)
    
    #plot outliers against original data
    from matplotlib import cm
    cmap = cm.get_cmap('Set1')
    course_grades.plot.scatter(x='course_views',y='OVERALL_GRADE', c=clusters, cmap=cmap, colorbar = False)
    
    clusters = pd.DataFrame(clusters,columns=['Outlier'])
    for index, value in enumerate(clusters['Outlier']):
        if(value == -1):
            #find the outlier & set it to mean value of the grade
            print(course_grades.loc[index]['course_views'])
            
            grade = course_grades.loc[index]['OVERALL_GRADE']
            
            print(course_grades.loc[course_grades['OVERALL_GRADE'] == grade].mean(axis = 0, skipna = True)['course_views'])
            ##set course_views to mean of values for the specific grade
            course_grades.at[index,'course_views'] = course_grades.loc[course_grades['OVERALL_GRADE'] == grade].mean(axis = 0, skipna = True)['course_views']
            course_views.at[index] = course_grades.loc[course_grades['OVERALL_GRADE'] == grade].mean(axis = 0, skipna = True)['course_views']
                                                                       
    g = sns.lmplot(x='course_views',y='OVERALL_GRADE',data=course_grades, sharex=False, sharey=False)
    g.set(ylim=(0, 5))
    
    course_corr, _ = pearsonr(course_views, grades)
    print("Pearson course views linear relation with grade after", course_corr,"\n")
    
test_course_views(students_df)

#%% Outlier Replacement in students_df
def replace_outliers(students_df):
    ##Remove all outliers
    ###for every column in students_df
    features = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE'], axis=1)
    for feature in features: 
        course_feature = students_df[feature];
        grades = students_df['OVERALL_GRADE']
        course_grades = pd.concat([course_feature,grades],axis='columns')
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler() 
        scaled = scaler.fit_transform(course_grades)
        #print(course_grades)
        ##DBSCAN detects outliers using a clustering method
        from sklearn.cluster import DBSCAN
        ##lower eps will make more clusters
        ##min_samples is number of dimentions + 1, a single feature is extracted and compared to grade result, therefore two dimentions are used
        ##n_jobs will use concurrent processing when set to -1
        ##euclidean distance is better performing in low dimensional datasets
        outlier_detection = DBSCAN(eps = .10, metric="euclidean", min_samples =3 , n_jobs = -1)
        clusters = outlier_detection.fit_predict(scaled)
        
        clusters = pd.DataFrame(clusters,columns=['Outlier'])
        for index, value in enumerate(clusters['Outlier']):
            if(value == -1):
                #find the outlier & set to MEAN value where overall_grade col == grade
                grade = course_grades.loc[index]['OVERALL_GRADE']
                ##set course_views to mean of values for the specific grade
                students_df.at[index,feature] = students_df.loc[students_df['OVERALL_GRADE'] == grade].mean(axis = 0, skipna = True)[feature]
        
replace_outliers(students_df)
    
#%% Find the corelation for every feature
def get_correlations(students_df):
    features = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE'], axis=1)
    #for each feature in students_df
    corelations = list()
    grade = students_df['OVERALL_GRADE']
    for feature in features:
        
        feature_extracted = students_df[feature]
        if((students_df[feature] == 0).all()):
             corr=0
        else:
            corr, _ = pearsonr(feature_extracted,grade)
            #print("Feature:",feature,"\n")
            #print("Corellation:",corr,"\n")
            #print("----------------------------------")
           
        corelations.append([feature,corr])
    
    corelations = pd.DataFrame(corelations)
    corelations.columns = ['Feature', 'Corelation']
    
    corelations = corelations.sort_values('Corelation', ascending=False)

    return corelations

corelations = get_correlations(students_df)
#print(corellations.head(20))
#corellations.head(20).to_csv('top20Activities.csv')

#%% Get the columns from correlations and select features from student df
topActivities = corelations.head(20)
topList = list()

for val in topActivities['Feature']:
    topList.append(val)

topList.append('OVERALL_GRADE')
features = students_df[topList]
#print(len(features.index))

#%% Use the Random Forest classifier
X = features.iloc[:,0:20].values
y = features.iloc[:, 20].values
def rf_predict():
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {
                     'n_estimators': [5, 10, 15, 20, 100, 200],
                     'max_depth': [2, 5, 7, 9, 10, 11]
                 }
    
    
    classifier = RandomForestClassifier()
    grid_rf = GridSearchCV(classifier, param_grid, cv=2)
    grid_rf.fit(X_train, y_train)
    
    #classifier = RandomForestClassifier(n_estimators=200)
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    
    print("Best: %f using %s" % (grid_rf.best_score_, grid_rf.best_params_))
    #print("Grid scores \n", grid_clf.cv_results_)
    
    print('Test Accuracy: %.3f' % grid_rf.score(X_test, y_test),"\n")
    rfResult = grid_rf.score(X_test, y_test) 
    
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    #print("Random Forest Accuracy: ",accuracy_score(y_test, y_pred))
    
    #score=cross_val_score(classifier,X_test,y_test,cv=5)
    #print(score)
rf_predict()
    
#%% Use Stratified KFold without GridCV
def strat_rf_predict():
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {
                     'n_estimators': [5, 10, 15, 20, 100, 200],
                     'max_depth': [2, 5, 7, 9, 10, 11]
                 }
    
    
    classifier = RandomForestClassifier()
    grid_rf = GridSearchCV(classifier, param_grid, cv=2)
    grid_rf.fit(X_train, y_train)
    
    #classifier = RandomForestClassifier(n_estimators=200)
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    
    print("Best: %f using %s" % (grid_rf.best_score_, grid_rf.best_params_))
    #print("Grid scores \n", grid_clf.cv_results_)
    
    print('Test Accuracy: %.3f' % grid_rf.score(X_test, y_test),"\n")
    rfResult = grid_rf.score(X_test, y_test) 
    
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    #print("Random Forest Accuracy: ",accuracy_score(y_test, y_pred))
    
    #score=cross_val_score(classifier,X_test,y_test,cv=5)
    #print(score)
#%% Use the Random Forest classifier
def two_col_kmeans_clustering(X,y):
    ##x and y are numpy arrays
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    X = np.vstack((X, y)).T
    #print(X[:,0])
    #print(X[:,1])
    
    plt.scatter(X[:,0],X[:,1], label='True Position')
    plt.show()
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_)
    plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
#%% SVM 
def svm_predict():
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    svcClassifier = SVC(gamma='scale')
    
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    
    ## 2 fold as there are only two members of the class 0 in y_train
    grid_svc = GridSearchCV(svcClassifier, param_grid, cv=2)
    grid_svc.fit(X_train, y_train)
    grid_svc.best_params_
    
    print("Best: %f using %s" % (grid_svc.best_score_, grid_svc.best_params_))
    print('Test Accuracy: %.3f' % grid_svc.score(X_test, y_test))
    svcResult = grid_svc.score(X_test, y_test)
    
    #print(score)
    #print("Precision:",metrics.precision_score(y_test, y_pred))
    
    # Model Recall: what percentage of positive tuples are labelled as such?
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    #from sklearn.svm import SVC
    
    course_views_numpy = students_df['Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence'].to_numpy()
    two_col_kmeans_clustering(course_views_numpy,y)
    
svm_predict()