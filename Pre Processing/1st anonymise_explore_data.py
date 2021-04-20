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

logs_df = pd.read_csv('files/allLogs.csv')

## found each staff member by using logs_df['User full name'].unique()
## staff members do not have a matriculation number
staffNames = ['Adam Lyons','Yann Savoye','Fee Mathieson',
              'Nirmalie Wiratunga','Steven Rae','John Isaacs',
              '-', 'Kit-ying Hui', 'Mark Zarb','Rachael Sammon',
              'Fiona Caldwell','Susan Frost','Sadiq Sani','Ailsa McWhirter']

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
def activity_access_plot(activities,counts):
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
logs_df = logs_df.drop(columns=['User full name', 'Affected user', 'Origin', 'IP address'], axis=1)
users_df = users_df.drop(columns=[0], axis=1)

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
        datesplit = (val.split(',',1))
        date = datesplit[0]
        date_obj = date.split('/',2)
        
        d = datetime.date(2000+int(date_obj[2]), int(date_obj[1]), int(date_obj[0]))
        ##if date is less than the courework 2 deadline
        if(d <= datetime.date(2018,11,10)):
           time_list.append(d)
        del idx, val,datesplit,date,date_obj,d    
    return time_list

course_views = get_course_views(course_views_logs)

#get all view counts per day
time_count_df = pd.DataFrame.from_dict(Counter(course_views), orient='index').reset_index()
time_count_df = time_count_df.rename(columns={'index':'time', 0:'count'})

del course_views

#plot number of course views
date_plot(time_count_df, title='Number of course views over time')

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
    if (idx !=0):
        counts = df['count'].to_numpy().sum()
        name = df['name'].unique()
        event_list.append([name[0],counts])
        #print(name[0])
        del counts,name
    del idx,df
del event_views_dfs    

activities = list()
counts = list()

for idx,var in enumerate(event_list):
    activities.append(var[0])
    counts.append(var[1])

#plot number of course views
activity_access_plot(activities,counts)
