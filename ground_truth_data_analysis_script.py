
import pandas as pd
import plotly.graph_objects as go

four_row_data = pd.read_csv('Soybean_Paper_Raw_Data_4row_380.csv')

(four_row_data['average_pod_count']*10)*2/1000
four_row_data['predicted_pod_weight'] = (four_row_data['average_pod_count']*10)*2/1000

four_row_data['deviation'] = four_row_data['predicted_pod_weight']- four_row_data['wt'] 



layout = go.Layout(
                legend=dict(
                orientation="h")
                )

fig = go.Figure(layout=layout)

fig.add_trace(go.Scatter(x=[_ for _ in range(len(four_row_data['wt']))], y=four_row_data['predicted_pod_weight'],
                    mode='markers',
                         marker=dict(color='rgb(0, 86, 67)'),
                    name='predicted'))
fig.add_trace(go.Scatter(x=[_ for _ in range(len(four_row_data['wt']))], y=four_row_data['wt'],
                    mode='markers',
                         marker=dict(color='rgb(255, 200, 46)'),
                    name='wt',
                        ))
fig.add_trace(go.Scatter(x=[_ for _ in range(len(four_row_data['wt']))], y=[2.53 for _ in range(len(four_row_data['wt']))],
                    mode='lines',
                         marker=dict(color='rgb(0, 86, 67)'),
                    name='trend',
                        ))
fig.add_trace(go.Bar(x=[_ for _ in range(len(four_row_data['wt']))],y =four_row_data['deviation'], name='deviation', 
                     marker_color='rgb(100, 86, 67)'))

fig.update_layout(
    xaxis=dict(
        title='Experimental plots [no]',
        showline=True,
        showgrid=False,
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
        title='Soybean yield [kg]',
        showgrid=True,
        showline=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
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

fig.update_layout(legend=dict(
    title="CONV 4-ROW (UT Exp 20)"
))
fig.update_layout(barmode='stack')

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.04,
    xanchor="left",
    x=0.06
))


# fig = px.scatter(four_row_data, x=[_ for _ in range(len(four_row_data['wt']))], y="wt", trendline="ols")
fig.show()