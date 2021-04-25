# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:29:38 2021

@author: Ben
"""


import pandas as pd

logs_df = pd.read_csv('student_logs.csv')
activities_df = pd.read_csv('topActivities.csv')
users_df = pd.read_csv('users.csv')
users_df.rename( columns={'Unnamed: 0':'id'}, inplace=True )

for index, row in activities_df.iterrows():
    #print(row['Resource'])
    users_df.insert(index,row['Resource'],'')
    
    
#for every student find their views for the activity
for index, row in users_df.iterrows():
    #print(index,row)
    print(index,"\n")
    #for every column in the row
    for idx, col in enumerate(row):
        #locate the student and the activity
        student = index
        if(idx < len(activities_df.index)):
            activity = activities_df.loc[idx]['Resource']
            #get logs where student and activity occur
            log = logs_df.loc[(logs_df['id'] == student) & (logs_df['Event context'] == activity)]
            
            print(len(log.index), activity)
            #users_df.insert(student,row[activity],len(log.index))
            #locate student and and activity in users_df
            users_df.loc[(users_df['id'] == student),activity] = len(log.index)
            


export_df = users_df[['id','User full name','early','ontime','late']]
export_df = pd.merge(export_df, users_df, left_on='id', right_on='id', how='left').drop(['id','User full name_y','early_y','ontime_y','late_y'], axis=1)
export_df.to_csv('users.csv', sep=',', encoding='utf-8')
