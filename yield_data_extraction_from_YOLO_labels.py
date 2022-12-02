import pandas as pd
from tqdm import tqdm
import os
from glob import glob

import plotly.express as px # optional

plot_id = []


# FnLr_total_pod_count = []
# FnLr_average_pod_count = []

main_dict = {}

def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for d in dirs:
            plot_id.append(d)

my_files_list = list_files('E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\Soybean_trial_plot_data\\filed_trials_2022\\PR_4_Row\\', 'txt')


filepath = 'E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\Soybean_trial_plot_data\\filed_trials_2022\\PR_4_Row\\'


c = 0
for plot in tqdm(plot_id):
    annotation_files = []
    file_name = []
    pod_count =[]
    annotation_files = glob(filepath+'\\'+plot+'\\'+'*.txt') # no of images in each plot
    no_images = len(annotation_files)
    
    total_pod_count = 0 # count all the pods in a folder/plot/variety
    for a in range(len(annotation_files)):
        file_name.append(os.path.basename(annotation_files[a]))
        with open(annotation_files[a], 'r+') as annots:
            entries = annots.readlines() # equal to no of pods counted
            pod_count.append(len(entries))
            total_pod_count+=len(entries)
            annots.close()

        c+=1
    try:
        avg = total_pod_count/no_images
    except Exception:
        avg = 0
    main_dict[eval(plot)]=[no_images,total_pod_count, avg]

data = pd.DataFrame.from_dict(main_dict).T
data = data.sort_index(ascending=True)
data.columns = ['no_of_images','total_pod_count','average_pod_count']

data.to_csv('PR_4_Row.csv')
print(data)

# vis = (data['average_pod_count'].values.reshape((10, 20))) # Conv_2_Row.csv
vis = (data['average_pod_count'].values.reshape((19, 20))) # Conv_UTs_Expt20.csv
# vis = (data['average_pod_count'].values.reshape((9, 20))) # PR_2_Row.csv
# vis = (data['average_pod_count'].values.reshape((3, 20))) # PR_4_Row.csv

fig = px.imshow(vis, text_auto=True, color_continuous_scale=px.colors.diverging.RdYlGn)
fig.show() # config=config