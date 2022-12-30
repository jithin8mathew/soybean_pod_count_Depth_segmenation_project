#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# In[2]:


data = pd.read_csv('Re Soybean ground truth yield data\\2022 CASSELTON CONV 4-ROW (UT Exp 20).csv')


# In[3]:


data


# In[25]:


# convert plot weight into numpy array
# get the average weight of all the pods
weight_data = np.array(data['wt'])
average_weight = np.average(weight_data)
average_weight

# 19*20


# In[ ]:





# In[5]:


visualPlot = pd.read_csv('Re Soybean ground truth yield data\\Map Casselton Conv 4-Row (UT Exp 20).csv',header=None)


# In[6]:


# drop extra field information such as plot labels and arrows
visualPlot = visualPlot.drop(columns=[0, 21])
visualPlot = visualPlot.drop([19,20,21])


# In[7]:


# convert dataframe into array and flatten the array 
visual_data_np = np.array(visualPlot)
visual_data_np = (visual_data_np.ravel())


# In[8]:


# get the index of 7-0 or empty rows 
fill_rows = np.where(visual_data_np == '7-0')
fill_rows[0]


# In[26]:



# insert average value into indexes with 7-0
# weight_data = np.insert(weight_data, fill_rows, [str(average_weight)*len(weight_data)])
weight_data = np.insert(weight_data, list(fill_rows[0]), [average_weight for _ in range(len(fill_rows[0]))])


# In[27]:


len(weight_data)
negative_values = np.where(weight_data < 0)
print(negative_values)
# weight_data = np.place(weight_data, list(negative_values[0]), [average_weight for _ in range(len(negative_values[0]))])
weight_data[weight_data<0] = average_weight
# weight_data = np.place(weight_data, negative_values[0], average_weight)
print(len(weight_data))


# In[28]:


# reshape the ground_truth yield data to rows and columns based on the plot map
z = weight_data.reshape(19, 20)


# In[ ]:





# In[48]:


# plot a heatmap
# fig = go.Figure(data=go.Heatmap(z))
fig = px.imshow(np.round(z, 2))

fig = go.Figure(data=go.Heatmap(
                   z=z,
    text2 =z.astype(str)))

# fig.update_layout(margin = dict(t=200,r=200,b=200,l=200),
#     showlegend = False,
#     width = 700, height = 700,
#     autosize = True )
fig.show()


# In[49]:


import plotly.figure_factory as ff


# In[51]:


fig = ff.create_annotated_heatmap(np.round(z, 2))
fig.show()

