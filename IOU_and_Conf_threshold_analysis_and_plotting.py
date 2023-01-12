#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.graph_objects as go
import numpy as np

data = pd.read_csv('IOU_testing_YOLOv7_35k_output.csv',header=None)
precision = data[3]
recall = data[4]
map50 = data[5]
map5095 = data[6]

title = 'IOU confidence testing'

fig = go.Figure()

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision)-1)], y=data[3], name='precision',
                         line=dict(color='rgb(0, 86, 67)', width=3)))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision)-1)], y=data[4], name='recall',
                         line=dict(color='rgb(237,173,8)', width=3)))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision)-1)], y=data[5], name='mAP@50',
                         mode= 'markers+lines',
                         marker=dict(color='rgb(0, 86, 67)')))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision)-1)], y=data[6], name='mAP@50:95',
                         mode= 'markers+lines',
                         marker=dict(color='rgb(237,173,8)')))

fig.update_layout( xaxis_title='IOU threshold',
                   yaxis_title='Percentage')

fig.add_annotation(x=data[3].idxmax(), y=max(data[3]),
            text=str(round(max(data[3])*100,3)),
            font=dict(
                color='rgb(0, 86, 67)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data[4].idxmax(), y=max(data[4]),
            text=str(round(max(data[4]),3)*100),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data[5].idxmax(), y=max(data[5]),
            text=str(round(max(data[5])*100,5)),
            font=dict(
                color='rgb(0, 86, 67)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=data[6].idxmax(), y=max(data[6]),
            text=str(round(max(data[6])*100,5)),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)


fig.update_layout(
    xaxis=dict(
        showline=True,
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

fig.update_xaxes(
    ticktext=[str(_) for _ in range(20,96,5)],
    tickvals=[_ for _ in range(0,76,5)],
)

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.79
))

fig.show()

print('max mAP0.5 at')
data[5].idxmax()+20, max(data[5]*100)

print('max mAP0.5:95 at')
data[6].idxmax()+20, max(data[6]*100)

print('precision')
data[3].idxmax()+20, max(data[3])

print('recall')
data[4].idxmax()+20, max(data[4])

conf_data = pd.read_csv('Conf_testing_YOLOv7_35k.csv',header=None)
precision = conf_data[3]
recall = conf_data[4]
map50 = conf_data[5]
map5095 = conf_data[6]

fig = go.Figure()

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision))], y=conf_data[3], name='precision',
                         line=dict(color='rgb(0, 86, 67)', width=3)))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision))], y=conf_data[4], name='recall',
                         line=dict(color='rgb(237,173,8)', width=3)))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision)-1)], y=conf_data[5], name='mAP@50',
                         mode= 'markers+lines',
                         marker=dict(color='rgb(0, 86, 67)')))

fig.add_trace(go.Scatter(x=[_ for _ in range(len(precision))], y=conf_data[6], name='mAP@50:95',
                         mode= 'markers+lines',
                         marker=dict(color='rgb(237,173,8)')))

fig.update_layout( xaxis_title='Confidence threshold',
                   yaxis_title='Percentage')

fig.add_annotation(x=conf_data[3].idxmax(), y=max(conf_data[3]),
            text=str(round(max(conf_data[3])*100,3)),
            font=dict(
                color='rgb(0, 86, 67)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=conf_data[4].idxmax(), y=max(conf_data[4]),
            text=str(round(max(conf_data[4]),3)*100),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=conf_data[5].idxmax(), y=max(conf_data[5]),
            text=str(round(max(conf_data[5])*100,5)),
            font=dict(
                color='rgb(0, 86, 67)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)

fig.add_annotation(x=conf_data[6].idxmax(), y=max(conf_data[6]),
            text=str(round(max(conf_data[6])*100,5)),
            font=dict(
                color='rgb(237,173,8)',
                size=12
            ),
            showarrow=True,
            arrowhead=2)


fig.update_layout(
    xaxis=dict(
        showline=True,
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

fig.update_xaxes(
    ticktext=[str(_) for _ in range(10,91,5)],
    tickvals=[_ for _ in range(0,91,5)],
)

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.79
))

fig.show()

# steep change at 34

