# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:38:48 2021

@author: Ben
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import Counter

logs_df = pd.read_csv('files/allLogs2.csv')
unique = logs_df['User full name'].unique()

cw1Time = datetime.date(2019,10,25)
cw2Time = datetime.date(2019,12,4)
startYear = datetime.date(2019,8,1)
## found each staff member by using logs_df['User full name'].unique()
## staff members do not have a matriculation number
staffNames = ['Adam Lyons','Yann Savoye','Fee Mathieson',
              'Nirmalie Wiratunga','Steven Rae','John Isaacs',
              '-', 'Kit-ying Hui', 'Mark Zarb','Rachael Sammon',
              'Fiona Caldwell','Susan Frost','Sadiq Sani','Ailsa McWhirter',
              'KYLE MARTIN (1106883)', 'CampusMoodle', 'David Dixon','Ikechukwu Nkisi-Orji',
              'David Corsar','Anjana Wijekoon','Ashleigh Henderson','Kyle Martin','Isobel Gordon','Colin Beagrie',
              'Rachael Sammon as KIERAN MCKENZIE (1806221)','Carlos Moreno-Garcia','Ashleigh Henderson',
              'Steven Rae as YASSEEN AHMANACHE (1603243)'
              ]

# Draw Plot for dates
def date_plot(df, title="", xlabel='Date', ylabel='Value', dpi=100):
    plot_data = np.asarray(df)
    x,y = plot_data.T
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.xticks(rotation=90)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

#draw plot for activities accessed
def activity_access_plot(event_list):
    activities = list()
    counts = list()

    for idx,var in enumerate(event_list):
        activities.append(var[0])
        counts.append(var[1])
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
     
    # Horizontal Bar Plot
    ax.barh(activities, counts)
     
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
     
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
     
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
     
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
     
    # Show top values 
    ax.invert_yaxis()
     
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')
     
    # Add Plot Title
    ax.set_title('No of access per course activity up to 10th November',
                 loc ='left', )
     
    # Show Plot
    plt.show()
    
##get a date from moodle timestamp
def get_Date(datet):    
    datesplit = (datet.split(',',1))
    date = datesplit[0]
    date_obj = date.split('/',2)
    
    d = datetime.date(2000+int(date_obj[2]), int(date_obj[1]), int(date_obj[0]))
    return d


users_list = logs_df['User full name']

#Drop staff members from csv file
#Drop staff members from users list
for idx, val in enumerate(users_list):
    if (val in staffNames):
        ##print(idx,val)
        # Filter the data accordingly.
        logs_df = logs_df[logs_df['User full name'] != val]
        users_list.drop(index=idx, axis=0, inplace=True)

#Keep only unique users_list
users_list = users_list.unique()

#insert an anonymous id for each student
logs_df.insert(0,'anonymous_id','')
users_df = pd.DataFrame(users_list)


for idx, val in enumerate(logs_df['User full name']):
    #print(idx, val)
    #get current student
    curr = users_df[users_df[0] == val]
    curr = int(curr.index.values)
    
    #print(curr)
    #set anonymous id
    logs_df.iloc[idx,0] = curr
    del curr
    
#Drop identifying and unused    
#logs_df = logs_df.drop(columns=['User full name', 'Affected user', 'Origin', 'IP address'], axis=1)
#users_df = users_df.drop(columns=[0], axis=1)

#delete staff names
del staffNames
del val
del users_list

#get every event context
event_ctx = logs_df['Event context'].unique()

#get all course_views
course_views_logs = logs_df[logs_df['Event name'] == 'Course viewed']

def get_course_views(df):
    time_list = list()
    
    for idx, val in enumerate(df['Time']):
        d = get_Date(val)
        ##if date is less than the courework 2 deadline
        if((d >= startYear ) and (d <= cw2Time)):
           time_list.append(d)
        del idx, val, d    
    return time_list

course_views = get_course_views(course_views_logs)

#get all view counts per day
time_count_df = pd.DataFrame.from_dict(Counter(course_views), orient='index').reset_index()
time_count_df = time_count_df.rename(columns={'index':'time', 0:'count'})

del course_views

#plot number of course views
#date_plot(time_count_df, title='Number of course views over time')

#get a count of every course event
event_views_dfs = list()

for idx,val in enumerate(event_ctx):
    event_views_df = logs_df[logs_df['Event context'] == val]
    event_views = get_course_views(event_views_df)
    if event_views:
        event_time_count_df = pd.DataFrame.from_dict(Counter(event_views), orient='index').reset_index()
        event_time_count_df = event_time_count_df.rename(columns={'index':'time', 0:'count'})
        event_time_count_df.insert(0,'name',val)
        event_views_dfs.append(event_time_count_df)
        del event_time_count_df
    del idx,val,event_views_df,event_views
    
#summarise the counts of each course
event_list = list()
for idx,df in enumerate(event_views_dfs):
    #if (idx !=0):
    counts = df['count'].to_numpy().sum()
    name = df['name'].unique()
    event_list.append([name[0],counts])
        #print(name[0])
    del counts,name
    del idx,df
del event_views_dfs    


#plot number of course views
#activity_access_plot(event_list)

#find the top activities accessed
def Nmaxelements(lists, N): 
    final_list = []
    names = [item[0] for item in lists]
    counts = [item[1] for item in lists]
    
    for i in range(0, N):  
        max1 = 0
        max1name=''
        for j in range(len(counts)):
            #curr = names[j].split(':',1)
            #if (curr[0] != 'Assignment' and curr[0] != 'Forum'):
            if counts[j] > max1: 
                max1 = counts[j]; 
                max1name=names[j]
        counts.remove(max1);
        names.remove(max1name);
        final_list.append([max1name,max1]) 
          
    return(final_list) 
    
topactivities = Nmaxelements(event_list, len(event_list)-1)

#activity_access_plot(topactivities)
users_df.insert(1,'late',0)
users_df.insert(2,'ontime',0)
users_df.insert(3,'early',0)
users_df.rename( columns={0:'User full name'}, inplace=True )

TurnitinSubmissionNames = ['Turnitin Assignment: CW1 WRITEUP DROPBOX' , 'Turnitin Assignment: CW2 - WRITEUP Dropbox']

def TurnitinSubmissions(TurnitinSubmissionNames, users_df, logs_df, cw1Time, cw2Time):
    
    for value in TurnitinSubmissionNames:
        
        for index, row in users_df.iterrows():
            
            if(value == 'Turnitin Assignment: CW1 WRITEUP DROPBOX'):
                cwTime = cw1Time
            else:
                cwTime = cw2Time
                
            student = index
            assignment1 = logs_df.loc[(logs_df['anonymous_id'] == student) & (logs_df['Event context'] == value ) & (logs_df['Event name'] == 'Add Submission') ]
            #print(assignment1)
           
            status=''#empty
            if(assignment1.empty == False):
                for idx, val in assignment1.iterrows():
                    if(val['Time']):
                        timestamp = val['Time']
                        d = get_Date(timestamp)
                        
                        if ( (status != 'ontime' and d < cwTime) and (status != 'late' and d < cwTime) ):
                            print('early\n', d, val['Event context'], student)
                            status='early'
                        elif ( (status != 'late' and d == cwTime) ):
                            print('ontime\n', d, val['Event context'], student)
                            status='ontime'
                        else:
                            print('late\n', d, val['Event context'], student)
                            status='late'    
                        
                if(status == 'early'):        
                    users_df.early.iloc[index] += 1
                elif(status == 'ontime'):
                    users_df.ontime.iloc[index] += 1
                else:
                    users_df.late.iloc[index] += 1
      
    return users_df

users_df = TurnitinSubmissions(TurnitinSubmissionNames, users_df,logs_df, cw1Time, cw2Time)

AssignmentNames = ['Assignment: IPYNB Dropbox ZIPs', 'Assignment: CODE dropbox (not for Word docs!)']

def Assignments(AssignmentNames, users_df, logs_df, cw1Time, cw2Time):
    
    for value in AssignmentNames:
        
        for index, row in users_df.iterrows():
            
            if(value == 'Assignment: IPYNB Dropbox ZIPs'):
                cwTime = cw1Time
            else:
                cwTime = cw2Time
                
            student = index
            assignment1 = logs_df.loc[(logs_df['anonymous_id'] == student) & (logs_df['Event context'] == value ) & (logs_df['Event name'] == 'A submission has been submitted.' ) ]
            #print(assignment1)
           
            status=''#empty
            if(assignment1.empty == False):
                for idx, val in assignment1.iterrows():
                    if(val['Time']):
                        timestamp = val['Time']
                        d = get_Date(timestamp)
                        
                        if ( (status != 'ontime' and d < cwTime) and (status != 'late' and d < cwTime) ):
                            print('early\n', d, val['Event context'], student)
                            status='early'
                        elif ( (status != 'late' and d == cwTime) ):
                            print('ontime\n', d, val['Event context'], student)
                            status='ontime'
                        else:
                            print('late\n', d, val['Event context'], student)
                            status='late'    
                        
                if(status == 'early'):        
                    users_df.early.iloc[index] += 1
                elif(status == 'ontime'):
                    users_df.ontime.iloc[index] += 1
                else:
                    users_df.late.iloc[index] += 1
      
    return users_df

users_df = Assignments(AssignmentNames, users_df,logs_df, cw1Time, cw2Time)
##export to csv
logs_df.to_csv('anonymous_logs.csv', sep=',', encoding='utf-8', index=False)

topactivities = pd.DataFrame(topactivities, columns=["Resource", "Count"])
#print(topactivities)
topactivities.to_csv('topActivities.csv', sep=',', encoding='utf-8', index=False)

users_df.to_csv('anonymous_users.csv', sep=',', encoding='utf-8')


