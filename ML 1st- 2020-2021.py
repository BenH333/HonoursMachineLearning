# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:46:19 2021

@author: Ben
"""

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy import cov
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score

students_df = pd.read_csv('with_grades_df_2020_2021.csv')
students_df.rename( columns={'Unnamed: 0':'anonymous_id'}, inplace=True )
#%%##### Grade Distribution
def plot_grades(df):
    grades = df['OVERALL_GRADE'].value_counts()
    grades.plot.bar()
plot_grades(students_df)

course_views = students_df['Study Area: [Module 2020/2021] CM4107 - Full Time: Advanced Artificial Intelligence'];
gradeDictionary = {5:'A',4:'B',3:'C',2:'D',1:'E',0:'F'}

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
        #print("grade is F")
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
        #print("grade:",val,"\n")
        #print("current student df:",val,"\n")
        #print("current grade label:",5,"\n")
        students_df.at[idx, 'OVERALL_GRADE'] = 0
        #print(students_df.loc[idx]['OVERALL_GRADE'],"\n")
    elif(val == 5):
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
course_grades = pd.concat([course_views,students_df['OVERALL_GRADE']],axis='columns')
course_grades = course_grades.rename(columns={'Study Area: [Module 2020/2021] CM4107 - Full Time: Advanced Artificial Intelligence':'course_views'})

covariance = cov(course_grades['course_views'],course_grades['OVERALL_GRADE'])
print(covariance)
#positive covariance shows there is a linear relationship

#pearson shows there is a moderate linear relationship
course_corr, _ = pearsonr(course_views, course_grades['OVERALL_GRADE'])
print("pearson course views linear relation with grade", course_corr)

##Two variables may be related by a nonlinear relationship, such that the relationship is stronger or weaker across the distribution of the variables.
course_views_spearman, _ = spearmanr(course_grades['course_views'],course_grades['OVERALL_GRADE'])
print('Spearmans correlation: %.3f' % course_views_spearman)

#%% Find the corelation for every feature
def get_correlations(students_df):
    features = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE'], axis=1)
    #for each feature in students_df
    corelations = list()
    
    for feature in features:
        feature_extracted = students_df[feature]
        current_grade = students_df['OVERALL_GRADE']
        corr, _ = pearsonr(feature_extracted,current_grade)
        corelations.append([feature,corr])
    
    corelations = pd.DataFrame(corelations)
    corelations.columns = ['Feature', 'Corelation']
    
    corelations = corelations.sort_values('Corelation', ascending=False)

    return corelations

corelations = get_correlations(students_df)
#print(corellations.head(20))
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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Random Forest Accuracy: ",accuracy_score(y_test, y_pred))

#%% SVM 
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
