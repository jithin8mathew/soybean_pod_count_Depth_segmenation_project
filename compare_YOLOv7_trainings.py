#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.graph_objects as go

headers = ['sr.no','Epoch','GIoU_loss','Objectness_loss','Classification_loss', '6', '7', '8', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Val_GIoU', 'Val_objectness', 'Val_classification']
len(headers)

data = pd.read_csv('F:\\ABEN\\soybean_segmentation_project_2022\\soybean_with_background\\soybean_segmented_NBG\\results.csv', header = None)
data.columns = headers

data_BG = pd.read_csv('F:\\ABEN\\soybean_segmentation_project_2022\\soybean_with_background\\soybean_with_background\\results.csv', header = None)
data_BG.columns = headers

fig = go.Figure()

fig.add_trace(go.Scatter(x=[_ for _ in range(300)], y=data['mAP@0.5'][:100], name='No Background mAP@0.5',
                         line=dict(color='rgb(237,173,8)', width=3)))
fig.add_trace(go.Scatter(x=[_ for _ in range(300)], y=data['mAP@0.5:0.95'][:100], name = 'No Background mAP@0.5:0.95',mode='lines+markers',
                         line=dict(color='rgb(237,173,8)', width=3,  dash='dot' )))


fig.add_trace(go.Scatter(x=[_ for _ in range(300)], y=data_BG['mAP@0.5'][:100], name='Background mAP@0.5',
                         line = dict(color='rgb(15,133,84)', width=3)))
fig.add_trace(go.Scatter(x=[_ for _ in range(300)], y=data_BG['mAP@0.5:0.95'][:100], name='Background mAP@0.5:0.95',mode='lines+markers',
                         line = dict(color='rgb(15,133,84)', width=3, dash='dot')))

fig.update_layout( xaxis_title='Epochs',
                   yaxis_title='Percentage')

fig.update_layout(
    xaxis=dict(
        showline=True,
#         showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=True,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=True,
    plot_bgcolor='white'
)

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.2, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Background vs Non-Background training',
                              font=dict(family='Arial',
                                        size=20,
                                        color='rgb(15,133,84)'),
                              showarrow=False))

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.79
))

NB_M5 = max(data['mAP@0.5'][:100])
NB_M595 = max(data['mAP@0.5:0.95'][:100])

BG_M5 = max(data_BG['mAP@0.5'][:100])
BG_M595 = max(data_BG['mAP@0.5:0.95'][:100])


fig.update_layout(annotations=annotations)

fig.add_annotation(x=data['mAP@0.5'][:100].idxmax(), y=NB_M5,
            text=str(NB_M5),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data['mAP@0.5:0.95'][:100].idxmax(), y=NB_M595,
            text=str(NB_M595),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data_BG['mAP@0.5'][:100].idxmax(), y=BG_M5,
            text=str(BG_M5),
#             ax=60,
#             ay=30,
            font=dict(
                color='rgb(15,133,84)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data_BG['mAP@0.5:0.95'][:100].idxmax(), y=BG_M595,
            text=str(BG_M595),
            font=dict(
                color='rgb(15,133,84)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.show()

