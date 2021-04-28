# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:46:19 2021

@author: Ben
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
from matplotlib import cm
plt.rcParams.update({'font.size': 10})

students_df = pd.read_csv('with_grades_df_2020_2021.csv')
students_df.rename( columns={'Unnamed: 0':'anonymous_id'}, inplace=True )

gradeDictionary = {6:'A',5:'B',4:'C',3:'D',2:'E',1:'F',0:'NS'}

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
    if(val == 6):
        #print("grade is NS")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",5,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 0
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 5):
        #print("grade is F")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",4,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 0
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 4):
        #print("grade is E")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",3,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 1
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 3):
        #print("grade is D")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",2,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 2
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 2):
        #print("grade is C")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",1,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 3
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 1):
        #print("grade is B")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",0,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 4
        #print("new label:", students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 0):
        students_df.at[idx, 'OVERALL_GRADE'] = 5
        
#%% Course Views Logistic Regression & Outlier removal
def test_course_views(students_df):
    course_views = students_df['Study Area: [Module 2020/2021] CM4107 - Full Time: Advanced Artificial Intelligence'];
    grades = students_df['OVERALL_GRADE']
    course_grades = pd.concat([course_views,grades],axis='columns')
    
    course_grades = course_grades.rename(columns={'Study Area: [Module 2020/2021] CM4107 - Full Time: Advanced Artificial Intelligence':'course_views'})
    course_views = course_views.rename('course_views')
    stdcon = sns.regplot(x='course_views',y='OVERALL_GRADE', data=course_grades)
    course_corr, _ = pearsonr(course_views, grades)
    print("pearson course views linear relation with grade before", course_corr)
    
    #standard_dev = course_grades.std(axis = 0, skipna = True)
    #mean = course_grades.mean(axis = 0, skipna = True)
    
    
    scaler = MinMaxScaler() 
    scaled = scaler.fit_transform(course_grades)
    print(scaled)
    ##DBSCAN detects outliers using a clustering method
    
    outlier_detection = DBSCAN(eps = 0.5, metric="euclidean", min_samples = 5, n_jobs = -1)
    clusters = outlier_detection.fit_predict(scaled)
    
    #plot outliers against original data
    
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
    g.set(ylim=(0, 6))
    
    course_corr, _ = pearsonr(course_views, grades)
    print("pearson course views linear relation with grade after", course_corr)
    
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
        
        scaler = MinMaxScaler() 
        scaled = scaler.fit_transform(course_grades)
        #print(course_grades)
        ##DBSCAN detects outliers using a clustering method

        ##lower eps will make more clusters
        ##The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        ##n_jobs will use concurrent processing when set to -1
        ##euclidean distance is better performing in low dimensional datasets
        outlier_detection = DBSCAN(eps = .10, metric="euclidean", min_samples =3 , n_jobs = -1)
        clusters = outlier_detection.fit_predict(scaled)
        
        clusters = pd.DataFrame(clusters,columns=['Outlier'])
        for index, value in enumerate(clusters['Outlier']):
            if(value == -1):
                #find the outlier & set to MEAN value where overall_grade col == grade
                grade = course_grades.loc[index]['OVERALL_GRADE']
                #print("before ", students_df.at[index,feature])
                ##set course_views to mean of values for the specific grade
                students_df.at[index,feature] = students_df.loc[students_df['OVERALL_GRADE'] == grade].mean(axis = 0, skipna = True)[feature]
                #print("after ",-students_df.at[index,feature])
replace_outliers(students_df)

#%% Find the corelation for every feature
def get_correlations(students_df):
    features = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE'], axis=1)
    #for each feature in students_df
    corelations = list()
    grade = students_df['OVERALL_GRADE']
    for feature in features:
        
        feature_extracted = students_df[feature]
        #check if current feature is empty
        if((students_df[feature] == 0).all()):
             corr=0
        else:
            corr, _ = pearsonr(feature_extracted,grade)
        #print("Feature:",feature,"\n")
        #print("Corellation:",corr,"\n")
       
        corelations.append([feature,corr])
    
    corelations = pd.DataFrame(corelations)
    corelations.columns = ['Feature', 'Corelation']
    
    corelations = corelations.sort_values('Corelation', ascending=False)

    return corelations

corelations = get_correlations(students_df)
#print(corelations.head(20))
#corelations.head(20).to_csv('top20Activities.csv')

#%% Get the columns from correlations and select features from student df
topActivities = corelations.head(20)
topList = list()

for val in topActivities['Feature']:
    topList.append(val)

topList.append('OVERALL_GRADE')
features = students_df[topList]
#print(len(features.index))

#%% Use the Random Forest classifier
#X = features.iloc[:,0:20].values
#y = features.iloc[:, 20].values

def strat_rf_predict(X,y):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    #print(skf.get_n_splits(X, y))
    
    param_grid = {
                     'n_estimators': [5, 10, 15, 20, 100],
                     'max_depth': [2, 5, 7, 9, 10, 11]
                 }

    sc = StandardScaler()
    random_forest_data=list()
    for train_index, test_index in skf.split(X, y):
        data=list()
        #test, train split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(y_train)
        
        #standard scalar
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
    
        rf = RandomForestClassifier()
        best_score=0
        #hyperparameter tuning
        #https://stackoverflow.com/questions/34624978/is-there-easy-way-to-grid-search-without-cross-validation-in-python
        for g in ParameterGrid(param_grid):
            rf.set_params(**g)
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_test)
            
            score = accuracy_score(y_test, y_pred)
            # save if best
            if score > best_score:
                best_score = score
                #print(y_test)
                #print(y_pred)
                #best_recall = recall_score(y_test, y_pred,average='macro', zero_division=1)
                #best_precision = precision_score(y_test, y_pred,average='macro', zero_division=1)
                f1_macro = f1_score(y_test,y_pred, average='macro', zero_division=1)
                f1_micro = f1_score(y_test,y_pred, average='micro', zero_division=1)
                best_grid = g
        
        #print("RF Score: %0.5f" % best_score)
        #print("RF Recall: ", best_recall)
        #print("RF Precision: ", best_precision)
        #print("RF Macro F1: ", f1_macro)
        #print("RF Micro F1: ", f1_micro)
        #print("RF Grid:", best_grid)
        data = [best_score,best_grid,f1_micro,f1_macro]
        random_forest_data.append(data)
    df = pd.DataFrame(random_forest_data, columns=['Best Score','Best Grid','F1 Micro','F1 Macro'])
    return df
#random_forest_df = strat_rf_predict()     


#%% SVM 
def strat_svm_predict(X,y):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    svcClassifier = SVC(gamma='scale')
    
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    
    sc = StandardScaler()
    random_forest_data=list()
    for train_index, test_index in skf.split(X, y):
        data=list()
        #test, train split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #standard scalar
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
    
        svc = SVC(gamma='scale')
        best_score=0
        
        #hyperparameter tuning
        for g in ParameterGrid(param_grid):
            svc.set_params(**g)
            svc.fit(X_train,y_train)
            y_pred = svc.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            f1_micro = f1_score(y_test, y_pred, average='weighted')
            # save if best
            if score > best_score:
                best_score = score
                #best_recall = recall_score(y_test, y_pred,average='macro', zero_division=1)
                #best_precision = precision_score(y_test, y_pred,average='macro', zero_division=1)
                f1_macro = f1_score(y_test,y_pred, average='macro', zero_division=1)
                f1_micro = f1_score(y_test,y_pred, average='micro', zero_division=1)
                best_grid = g
                
        #print("SVC Score: %0.5f" % best_score) 
        #print("F1 Micro: %0.5f" % f1_micro)
        #print("SVC Grid:", best_grid)
        data = [best_score,best_grid,f1_micro,f1_macro]
        random_forest_data.append(data)
    df = pd.DataFrame(random_forest_data, columns=['Best Score','Best Grid','F1 Micro','F1 Macro'])
    return df
#svm_df = strat_svm_predict()

testFeatures = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,len(students_df.columns)-4]

def best_scores_with_feature_sets(testFeatures):
    all_svms=list();
    all_rfs=list();
    best_df = pd.DataFrame({'Number of Features':testFeatures})
    best_df["Random Forest"] = np.nan
    best_df["Support Vector Classifier"] = np.nan
    
    micro_df = pd.DataFrame({'Number of Features':testFeatures})
    micro_df["Random Forest"] = np.nan
    micro_df["Support Vector Classifier"] = np.nan
    
    print(best_df)
    for value in testFeatures:
        topActivities = corelations.head(value)
        topList = list()
        
        for val in topActivities['Feature']:
            topList.append(val)
        
        topList.append('OVERALL_GRADE')
        features = students_df[topList]
        
        X = features.iloc[:,0:value].values
        y = features.iloc[:, value].values
        
        random_forest_df = strat_rf_predict(X,y)  
        svm_df = strat_svm_predict(X,y)
        
        best_rf = random_forest_df.iloc[random_forest_df['Best Score'].idxmax()]
        best_svm = svm_df.iloc[svm_df['Best Score'].idxmax()]
        
        all_svms.append([value,best_svm])
        all_rfs.append([value,best_rf])
        
        best_df.loc[best_df['Number of Features'] == value, 'Random Forest'] = best_rf['Best Score'] 
        best_df.loc[best_df['Number of Features'] == value, 'Support Vector Classifier'] = best_svm['Best Score'] 
        
        micro_df.loc[micro_df['Number of Features'] == value, 'Random Forest'] = best_rf['F1 Micro']
        micro_df.loc[micro_df['Number of Features'] == value, 'Support Vector Classifier'] = best_svm['F1 Micro']
        
        
    best_df = best_df.melt('Number of Features', var_name='ML Algorithm on 2020/2021',  value_name='Accuracy')
    sns.set_style("darkgrid")
    sns.factorplot(x="Number of Features", y="Accuracy", hue='ML Algorithm on 2020/2021', data=best_df, palette=sns.color_palette('summer', n_colors=2))
    plt.show()
    micro_df = micro_df.melt('Number of Features', var_name='ML Algorithm on 2020/2021',  value_name='F1 Micro')
    sns.set_style("darkgrid")
    sns.factorplot(x="Number of Features", y="F1 Micro", hue='ML Algorithm on 2020/2021', data=micro_df, palette=sns.color_palette('summer', n_colors=2))
    plt.show()
    return all_rfs, all_svms, best_df
    
best_rf, best_svm, bestscores = best_scores_with_feature_sets(testFeatures)