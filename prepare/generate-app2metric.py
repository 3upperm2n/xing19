
# coding: utf-8

# In[74]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import os,sys
sys.path.append('../pycode')
from magic import read_trace, adjust_metric, gen_app2metric

import json,yaml,codecs

import numpy as np
import pandas as pd
import copy


# ### [1] read all the profling metrics on nvidia pascal arch 

# In[75]:


with open('../00_featsel/pascal_1080ti_metrics.json', 'r') as metricsFile:
     metricsAll = yaml.safe_load(metricsFile)
        
TargetMetrics = metricsAll['pascal_1080ti']        

featureDim = len(TargetMetrics)
print("Metrics on Pascal GPUs (1080Ti): {}".format(featureDim))


# In[76]:


#metricsAll


# ### [2] read scale for each feature metric

# In[77]:


#
# read the metrics_scale.jason
#
with open('../00_featsel/pascal_1080ti_metrics_scaler.json', 'r') as metricsFile:
     metrics_scale_dd = yaml.safe_load(metricsFile)


# In[78]:


# metrics_scale_dd


# In[79]:


metrics_all = []
for metric, _ in metrics_scale_dd.iteritems():
    metrics_all.append(metric)
    
# print metrics_all
print '\ntotal metrics number = %d' % len(metrics_all)


# ### [3] read app metrics,  convert the raw data into the same unit

# read all the metrics in **./metrics** folder, save the input in data frame

# In[80]:


metricsFolder = "../apps/metrics-all-1080ti/"


# Some file name examples:
# 
# * metrics_cudasdk_batchCUBLAS.csv
# * metrics_rodinia_hybridsort.csv

# In[81]:


appTrace = os.listdir(metricsFolder)

app_metrics_max_dd = {}

for currentFile in appTrace:
    appName = currentFile[8:][:-4]
    file_csv = metricsFolder + '/' + currentFile # csv link
    #print appName
    #print("current : {},  appName : {},  csv_loc : {}".format(currentFile, appName, file_csv))
    
    df_app = read_trace(file_csv) # read csv content

    appMetricMax_dd = {}
    
    for metric in metrics_all:
        try:
            df_metric = df_app.loc[df_app['Metric Name'] == metric]['Avg']
            m_list = [adjust_metric(metric, mVal) for _, mVal in df_metric.iteritems()]
            appMetricMax_dd[metric] = max(m_list)  # use the max() value for the current feature column
        except Exception as e:
            print e.message, e.args
            print('ERROR!! App = {}, Metric Name = {}'.format(appName, metric))
            sys.exit(1)

    app_metrics_max_dd[appName] = appMetricMax_dd  # update app metrics for current application


# In[82]:


#app_metrics_max_dd['cudasdk_reduction']


# ### [4 ]transform data dict to data frame (NOTE: this seems to be redundant !)

# In[83]:


print("Metrics on Pascal GPUs (1080Ti): {}".format(featureDim))


# In[84]:


featMatCols = ['AppName']         
featMatCols.extend(TargetMetrics)


# In[85]:


# application number
appNum = len(app_metrics_max_dd)
print "Total applications :  %d" % appNum


# In[86]:


#  appNum  x featureDim
df_app = pd.DataFrame(index=np.arange(0, appNum), columns=featMatCols)


# In[87]:


#
# export data to data frame, so that we can export to csv file easily
#
rowId = 0
for appName, metrics_dd in app_metrics_max_dd.iteritems():
    df_app.loc[rowId, 'AppName'] = appName # fill in kernel name 

    # add more metrics according to the column order
    for eachMetric in TargetMetrics:
        try:
            df_app.loc[rowId, eachMetric] = metrics_dd[eachMetric]
        except Exception as e:
            print e.message, e.args
            print('ERROR!! App = {}, Metric Name = {}'.format(appName, eachMetric))
            sys.exit(0)

    rowId += 1


# In[88]:


#df_app


# ### [5] apply minmax scaler

# In[89]:


# the scaling factors are stored in metrics_scale_dd

#df_app_scale = df_app.copy()
df_app_scale = copy.deepcopy(df_app)

#metrics_scale_dd

for metric in TargetMetrics:
    [x_min, x_max] = metrics_scale_dd[metric]  # read the scaler from dict for min and max value of the feature
    
    if x_max == x_min:
        x_range = 1e-6    # up-floor, avoid float division by zero
    else:
        x_range = x_max - x_min
    
    df_app_scale[metric] = df_app_scale[metric].apply(lambda x : (x - x_min) / x_range)


# In[90]:


df_app_scale


# In[91]:


# save dataframe to csv
#df_app_scale.to_csv('1080ti-featAll.csv', index=False, encoding='utf-8')

# save to dictionary
app2metric_featAll = gen_app2metric(df_app_scale)
np.save('app2metric_featAll.npy', app2metric_featAll)


# ### [6] feat9

# In[92]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat9.json", "r") as read_file:
    feat9_list = json.load(read_file)
print len(feat9_list)



#
# select the columns that match the feature list
#
sel_feats = feat9_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat9 = gen_app2metric(df_current)
np.save('app2metric_feat9.npy', app2metric_feat9)


# ### [7] feat12

# In[93]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat12.json", "r") as read_file:
    feat12_list = json.load(read_file)
print len(feat12_list)

#
# select the columns that match the feature list
#
sel_feats = feat12_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat12 = gen_app2metric(df_current)
np.save('app2metric_feat12.npy', app2metric_feat12)


# ### [8] feat14

# In[94]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat14.json", "r") as read_file:
    feat14_list = json.load(read_file)
print len(feat14_list)

#
# select the columns that match the feature list
#
sel_feats = feat14_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat14 = gen_app2metric(df_current)
np.save('app2metric_feat14.npy', app2metric_feat14)


# ### [9] feat18

# In[95]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat18.json", "r") as read_file:
    feat18_list = json.load(read_file)
print len(feat18_list)

#
# select the columns that match the feature list
#
sel_feats = feat18_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat18 = gen_app2metric(df_current)
np.save('app2metric_feat18.npy', app2metric_feat18)


# ### [10] feat26

# In[97]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat26.json", "r") as read_file:
    feat26_list = json.load(read_file)
print len(feat26_list)

#
# select the columns that match the feature list
#
sel_feats = feat26_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat26 = gen_app2metric(df_current)
np.save('app2metric_feat26.npy', app2metric_feat26)


# ### [11] feat42

# In[98]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat42.json", "r") as read_file:
    feat42_list = json.load(read_file)
print len(feat42_list)

#
# select the columns that match the feature list
#
sel_feats = feat42_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat42 = gen_app2metric(df_current)
np.save('app2metric_feat42.npy', app2metric_feat42)


# ### [12] feat64

# In[99]:


#
# read in the feature list
#
with open("../00_featsel/1080ti_PFA_feat64.json", "r") as read_file:
    feat64_list = json.load(read_file)
print len(feat64_list)

#
# select the columns that match the feature list
#
sel_feats = feat64_list

other_feats = [m for m in TargetMetrics if m not in sel_feats]

# double check
if len(TargetMetrics) <> (len(other_feats) + len(sel_feats)):
    print "The feats number does not match!"
else:
    print "Good job!"
    
df_current = copy.deepcopy(df_app_scale)
df_current.drop(other_feats, axis = 1, inplace=True)  # inplace drop columns not needed
#df_current.to_csv("1080ti-feat9.csv", index=False, encoding='utf-8')

#
# save df as dd, with appName as the key, and features in np array!
#
app2metric_feat64 = gen_app2metric(df_current)
np.save('app2metric_feat64.npy', app2metric_feat64)

