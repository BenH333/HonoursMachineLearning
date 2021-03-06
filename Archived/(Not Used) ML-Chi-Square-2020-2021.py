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

import scipy.stats as stats
from scipy.stats import chi2_contingency

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
    #stdcon = sns.regplot(x='course_views',y='OVERALL_GRADE', data=course_grades)
    course_corr, _ = pearsonr(course_views, grades)
    print("pearson course views linear relation with grade before", course_corr)
    
    
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
    #course_grades.plot.scatter(x='course_views',y='OVERALL_GRADE', c=clusters, cmap=cmap, colorbar = False)
    
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
                                                                       
    #g = sns.lmplot(x='course_views',y='OVERALL_GRADE',data=course_grades, sharex=False, sharey=False)
    #g.set(ylim=(0, 6))
    
    course_corr, _ = pearsonr(course_views, grades)
    print("pearson course views linear relation with grade after", course_corr)
    
#test_course_views(students_df)
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
    
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(X)
    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_)
    plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    
#%% Correlation testing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def chi_square_best(students_df):
    features_extracted = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2'], axis=1)
    
    # Create features and target
    X = features_extracted.iloc[:,0:len(features_extracted.columns)-1].values
    y = features_extracted.iloc[:, len(features_extracted.columns)-1].values
    x_labels = features_extracted.drop(['OVERALL_GRADE'], axis=1).columns.values
    
    # Convert to categorical data by converting data to integers
    X = X.astype(int)
    
    # Select two features with highest chi-squared statistics
    chi2_selector = SelectKBest(chi2, k=20)
    chi2_selector.fit(X, y)
    
    # Look at scores returned from the selector for each feature
    chi2_scores = chi2_selector.scores_, chi2_selector.pvalues_
    chi2_scores = pd.DataFrame(list(zip(x_labels,chi2_selector.scores_, chi2_selector.pvalues_)), columns=['label','score', 'pval'])
    chi2_scores
    
    # you can see that the kbest returned from SelectKBest 
    #+ were the two features with the _highest_ score
    kbest = chi2_selector.get_support()
    kbest = pd.DataFrame(kbest, columns=['best'])
    labels = pd.DataFrame(x_labels, columns=['label'])
    final = kbest.join(labels)
    final = final.loc[final['best']==True].drop(['best'],axis=1)
    return final

topActivities = chi_square_best(students_df)
topList = list()

for val in topActivities['label']:
    topList.append(val)

topList.append('OVERALL_GRADE')
features = students_df[topList]

#%% Use the Random Forest classifier
X = features.iloc[:,0:20].values
y = features.iloc[:, 20].values

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sc = StandardScaler()
def rf_predict():
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
        
    param_grid = {
                     'n_estimators': [5, 10, 15, 20, 100],
                     'max_depth': [2, 5, 7, 9, 10, 11]
                 }
    
    
    classifier = RandomForestClassifier()
    grid_rf = GridSearchCV(classifier, param_grid,cv=3)
    grid_rf.fit(X_train, y_train)
    
    #classifier = RandomForestClassifier(n_estimators=200)
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    
    print("RF Best: %f using %s" % (grid_rf.best_score_, grid_rf.best_params_))
    #print("Grid scores \n", grid_clf.cv_results_)
    
    print('RF Test Accuracy: %.3f' % grid_rf.score(X_test, y_test),"\n")
    rfResult = grid_rf.score(X_test, y_test) 
    
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    #print("Random Forest Accuracy: ",accuracy_score(y_test, y_pred))
    
    #score=cross_val_score(classifier,X_test,y_test,cv=5)
    #print(score)
#rf_predict()

##stratified kfold from https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb
def strat_rf_predict():
    skf = StratifiedKFold(n_splits=4)
    #print(skf.get_n_splits(X, y))
    
    param_grid = {
                     'n_estimators': [5, 10, 15, 20, 100],
                     'max_depth': [2, 5, 7, 9, 10, 11]
                 }

    sc = StandardScaler()
    
    for train_index, test_index in skf.split(X, y):
    
        #test, train split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
                best_grid = g
                
        print("RF Score: %0.5f" % best_score) 
        print("RF Grid:", best_grid)
        
#strat_rf_predict()        

#%% SVM 
def svm_predict():
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    
    
    svcClassifier = SVC(gamma='scale')
    
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    
    grid_svc = GridSearchCV(svcClassifier, param_grid,cv=3)
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
    
def strat_svm_predict():
    skf = StratifiedKFold(n_splits=4)
    
    svcClassifier = SVC(gamma='scale')
    
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    
    sc = StandardScaler()
    
    for train_index, test_index in skf.split(X, y):
    
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
            # save if best
            if score > best_score:
                best_score = score
                best_grid = g
                
        print("SVC Score: %0.5f" % best_score) 
        print("SVC Grid:", best_grid)
        
#strat_svm_predict()

course_views_numpy = students_df['Study Area: [Module 2020/2021] CM4107 - Full Time: Advanced Artificial Intelligence'].to_numpy()
#two_col_kmeans_clustering(course_views_numpy,y)

def kmeans_input_features(features):
    ##x and y are numpy arrays
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    y_data = pd.read_csv('with_grades_df_2020_2021.csv')
    y_data = y_data['OVERALL_GRADE']
    
    clustering_kmeans = KMeans(n_clusters=7, precompute_distances="auto", n_jobs=-1)
    features['Clusters'] = clustering_kmeans.fit_predict(features)
    features['Clusters'] = features['Clusters'] + 1
    features = features.drop(labels='OVERALL_GRADE', axis=1)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    
    principalDf = pd.DataFrame(data = principalComponents ,
                               columns = ['Principal Component 1', 'Principal Component 2'])
    
    finalDf = pd.concat([principalDf, y_data], axis = 1)
    colors = {0:'C0', 1:'C1', 2:'C2', 3:'C3', 4:'C4', 5:'C5', 6:'C6'}
    sns.set_style("darkgrid")
    sns.scatterplot(x="Principal Component 1", y="Principal Component 2", hue=features['Clusters'], data=finalDf, palette='viridis_r')
    plt.title('K-means Clustering with 2 dimensions')
    plt.show()
    
kmeans_input_features(features)

def pca_scatter(features):
    from sklearn.decomposition import PCA
    
    x_data = features.drop(labels='OVERALL_GRADE', axis=1)
    y_data = pd.read_csv('with_grades_df_2020_2021.csv')
    y_data = y_data['OVERALL_GRADE']
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    
    principalDf = pd.DataFrame(data = principalComponents ,
                               columns = ['Principal Component 1', 'Principal Component 2'])
    
    finalDf = pd.concat([principalDf, y_data], axis = 1)
    print(finalDf.head())
    
    colors = {'NS':'C0','F':'C1', 'E':'C2', 'D':'C3', 'C':'C4', 'B':'C5', 'A':'C6'}
    sns.set_style("darkgrid")
    sns.scatterplot(x="Principal Component 1", y="Principal Component 2", hue=finalDf['OVERALL_GRADE'],data=finalDf, palette='viridis_r')
    plt.title('Grade with Two Dimensions')
    plt.show()
    
    
pca_scatter(features)   