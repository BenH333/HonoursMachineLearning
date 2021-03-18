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
#dummies = pd.get_dummies(students_df['OVERALL_GRADE'])

#merged = pd.concat([students_df, dummies], axis='columns')
#del dummies

#merged.rename( columns=gradeDictionary, inplace=True )

grades = students_df['OVERALL_GRADE']
course_grades = pd.concat([course_views,grades],axis='columns')

course_corr, _ = pearsonr(course_views, grades)
print("pearson course views linear relation with grade", course_corr)
course_grades = course_grades.rename(columns={'Study Area: [Module 2019/2020] CM4107 - Full Time: Advanced Artificial Intelligence':'course_views'})

stdcon = sns.regplot(x='course_views',y='OVERALL_GRADE', data=course_grades)

#%% Chi Square
perc = (lambda col:col/col.sum())
index = [0,1,2,3,4,5]
stdval = pd.crosstab(index=course_grades.OVERALL_GRADE,columns=course_grades.course_views)
#stdval1 = np.log(stdval)
#stdval2 = stdval.apply(perc).reindex(index)

#print(stdval2)
#stdval2.plot.bar(colormap="PiYG_r", fontsize=20, figsize=(10,7))
#plt.title("Final grades by course views", fontsize=16)
#plt.ylabel("Percentage of log std count", fontsize=16)
#plt.xlabel("Final Grade", fontsize=16)
#plt.show()


#chi-sqaured test
stdfinalgrade = sm.stats.Table(stdval)
stdfinalrslt = stdfinalgrade.test_nominal_association()
print(stdfinalrslt.pvalue)

#%% Corellation testing
#features = merged.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE','A','B','C','D','E','NS',
#                        'Assignment: CM4107 Resit Coursework Dropbox - Due 14/08/2020 @ 4pm'], axis=1)
#corellations = list()
#for key,grade in gradeDictionary.items():
#for feature in features:
#    print("Feature ----------")
#    print(feature)
#    print(" Overall Grade ---------------")
#    print("corellation -----------------")
#    feature_extracted = merged[feature]
#    #current_grade = merged[grade]
#    corr, _ = pearsonr(feature_extracted,grades)
#    print(corr)
#    corellations.append([feature,corr])
#    print("----------------------------------")


from numpy import cov

covariance = cov(course_grades['course_views'],course_grades['OVERALL_GRADE'])
print(covariance)
#positive covariance shows there is some linear realtionship

course_views_pearson, _ = pearsonr(course_grades['course_views'],course_grades['OVERALL_GRADE'])
#pearson shows there is a moderate linear relationship

##Two variables may be related by a nonlinear relationship, such that the relationship is stronger or weaker across the distribution of the variables.
from scipy.stats import spearmanr
course_views_spearman, _ = spearmanr(course_grades['course_views'],course_grades['OVERALL_GRADE'])

#print('Spearmans correlation: %.3f' % spearman)

#%% Find the corellation for every feature
features = students_df.drop(['anonymous_id','COURSEWORK_1','COURSEWORK_2','OVERALL_GRADE',
                        'Assignment: CM4107 Resit Coursework Dropbox - Due 14/08/2020 @ 4pm'], axis=1)
#for each feature in students_df
corellations = list()
for feature in features:
    print("Feature ----------")
    print(feature)
    print(" Overall Grade ---------------")
    print("corellation -----------------")
    feature_extracted = students_df[feature]
    current_grade = students_df['OVERALL_GRADE']
    corr, _ = pearsonr(feature_extracted,current_grade)
    print(corr)
    corellations.append([feature,corr])
    print("----------------------------------")

