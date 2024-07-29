from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

import base64
import datetime
import io
import os

import pandas as pd
from statistics import mean 
from statistics import stdev

style_value_text = {'fontSize': 30, 'textAlign': 'center'}
style_value_value = {"textAlign": "center", "fontSize": 30}
style_value_figure = {'display': 'inline-block'}
style_value_global = {'fontSize': 15, 'textAlign': 'center'}

palette_1 = ["#40a49c", "#40bcb4", "#e8f4f4"]  
palette_2 = ["#40a49c", "#e8f4f4"]
palette_3 = ["#40a49c", "#e8f4f4"]
palette_4 = ["#40a49c", "#e8f4f4"]
palette_5 = ["#40a49c", "#e8f4f4"]
personal_palette = ["#40a49c", "#40bcb4", "#e8f4f4"]

def create_fig(metric, metric_format, title_1, title_2):
    #Add indicator
    fig = go.Figure(go.Indicator(
        mode = 'number',
        value = metric,
        number = {'valueformat': metric_format, 'font.size': 50},
        domain = {'x': [0, 1], 'y': [0, 0.65]}))

    #Add title
    fig.update_layout(title = {'text': '{}<br>{}'.format(title_1, title_2), 'x': 0.5, 'xanchor': 'center', 'y': 0.1, 'yanchor': 'top', 'font.size': 12},
                      paper_bgcolor='rgba(0,0,0,0)',
                      font_color='white',
                      autosize=False,
                      width=180,
                      height=180,
                      margin=dict(l=2, r=2, b=2, t=2))

    return fig

from scipy.stats import gaussian_kde
import numpy as np

def distribution_fig(metric, df_result):
    # read csv
    df = pd.read_csv(os.path.join('data', 'df_subset.csv'))

    data = df[metric].dropna()  # Ensure no NaN values

    # Create the figure
    fig = go.Figure()

    # Calculate the KDE
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    kde_values = kde(x_range)

    # Add line plot for the KDE
    fig.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name='Density', marker=dict(color='black')))

    # Add scatter plot for specific points
    y_value = kde(df_result[df_result['Metric'] == metric]['Value'])
    fig.add_trace(go.Scatter(x=df_result[df_result['Metric'] == metric]['Value'], y=y_value, mode='markers', marker=dict(color='red', size=10, line=dict(color='black', width=1)), name='Your Value'))
    
    # Add scatter plot for specific points
    for group in df_result.drop(['Metric', 'Value'], axis = 1).columns:
        group_data = df_result[df_result['Metric'] == metric][group]
        y_values = kde(group_data)
        fig.add_trace(go.Scatter(x=group_data, y=y_values, mode='markers', marker=dict(size=10, line=dict(color='black', width=1)), name=group))

    # Customize layout to remove grid
    fig.update_layout(
        xaxis_title=metric, 
        yaxis_title='Density', 
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),  # Remove x-axis grid
        yaxis=dict(showgrid=False),   # Remove y-axis grid
        font=dict(color='#000000')  # Make all other text black
    )

    return dcc.Graph(id="figure", figure=fig)

def arc_progression(arcs: list) -> dcc.Graph:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = list(range(len(arcs))),
        y = arcs,
        mode = 'lines+markers',
        name = 'Arc Progression',
        marker = dict(
            color = 'black',
            size = 10,
            line = dict(
                color = 'black',
                width = 1
            )
        )
    ))

    fig.update_layout(
        xaxis_title = 'Arc',
        yaxis_title = 'Sentiment',
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        height = 400,
        margin = dict(l = 20, r = 20, t = 20, b = 20),
        xaxis = dict(showgrid = False),
        yaxis = dict(showgrid = False),
        font = dict(color = '#000000')
    )

    return dcc.Graph(id = "arc_progression", figure = fig)


def value_boxes(column_name: str, value_name: str, df: pd.DataFrame, color: str) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{value_name}", style=style_value_text),
                html.Div(f"{df[df['Metric'] == column_name]['Value'].values[0].round(2)}", style=style_value_value),
                html.Div(f"(Bestsellers Mean {value_name} {df[df['Metric'] == column_name]['Mean_Bestsellers'].values[0].round(2)})", style=style_value_global) if not df[df['Metric'] == column_name]['Mean_Bestsellers'].isna().any() else None,
                html.Div(f"(Canonicals Mean {value_name} {df[df['Metric'] == column_name]['Mean_Canonicals'].values[0].round(2)})", style=style_value_global) if not df[df['Metric'] == column_name]['Mean_Canonicals'].isna().any() else None
            ])
        ] if column_name in df['Metric'].values else None, 
        style = {"backgroundColor": color, 'borderColor': 'black'})
    ], width = {'size': 3, 'offset': 2})

def value_boxes_1(value_name: str, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{value_name}", style=style_value_text),
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black'}|extra_style)
    ], width = location)

def value_boxes_new(column_name: str, color: str) -> dbc.Col:
    return dbc.Card([
            dbc.CardBody([
                html.Div(f"{column_name}", style=style_value_text),
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black'})

def value_boxes_2(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['Value'].values[0].round(2)}", style=style_value_value)
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)

def value_boxes_3(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['BESTSELLERS'].values[0].round(2)}", style=style_value_value)
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)

def value_boxes_4(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['CANON_ALL'].values[0].round(2)}", style=style_value_value)
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)

def value_boxes_1234(value_name: str, column_name: str, df: pd.DataFrame, color: str) -> dbc.Col:
    return dbc.Row([
            value_boxes_1(column_name, palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2(value_name, df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3(value_name, df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4(value_name, df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20})

def value_boxes_12(value_name: str, column_name: str, df: pd.DataFrame, color: str) -> dbc.Col:
    return dbc.Row([
            dbc.Col([value_boxes_new(column_name, palette_1[2])], width = {'size': 2, 'offset': 0}, align="center"),
            dbc.Col([value_boxes_new(df[df['Metric'] == value_name]['Value'].values[0].round(2), palette_1[2])], width = {'size': 2, 'offset': 4}, align="center"),
        ], style={"marginTop": 20, "marginBottom": 20})

def value_boxes_fig(value_name: str, column_name: str, df: pd.DataFrame, color: str) -> dbc.Col:
    return dbc.Row([
            dbc.Col([value_boxes_new(column_name, palette_1[2])], width = {'size': 2, 'offset': 0}, align="center"),
            dbc.Col([distribution_fig(value_name, df)], width = {'size': 9, 'offset': 1}),
        ], style={"marginTop": 0, "marginBottom": 0})

def value_boxes_arcs_fig(name: str, arcs: list[float]) -> dbc.Col:
    return dbc.Row([
            dbc.Col([value_boxes_new(name, palette_1[2])], width = {'size': 2, 'offset': 0}, align="center"),
            dbc.Col([arc_progression(arcs)], width = {'size': 9, 'offset': 1}),
        ], style={"marginTop": 0, "marginBottom": 0})

def metrics_explanation(metric_group: str, explanation: str, id_but: str, id_col) -> dbc.Row:
    return dbc.Row([
         dbc.Col([
              html.I(className="bi bi-caret-right-fill", n_clicks = 0, id=id_but, style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
              html.H5("Description of metrics", style={"display": "inline"}),
        ], width = {'size': 3, 'offset': 1}),
        dbc.Collapse(
            dbc.Card([
                html.H3(children=f"Explanation of {metric_group} Metrics", style = {'textAlign': 'left'}),
                html.Div(dcc.Markdown(explanation, dangerously_allow_html=True, link_target="_blank"))
            ], style = {'backgroundColor': palette_1[1], 'borderColor': 'black', 'padding': '10px'}), id=id_col, is_open=False),
        ])

def styl_func(style_df: pd.DataFrame, stylometrics_explanation_text: str, language: str) -> html.Div:
    column_mapping = {
        'word_count': 'Word Count',
        'average_wordlen': 'Word Length',
        'msttr': 'MSTTR',
        'average_sentlen': 'Sentence Length',
        'bzipr': 'bzipr',
        'word_entropy': 'Word Entropy',
        'bigram_entropy': 'Bigram Entropy'
    }

    content = [html.H2(children='Stylometrics', className="fw-bold text-white")]

    if language == 'english':
        for value_name in style_df['Metric']:
            if value_name in column_mapping:
                content.append(value_boxes_fig(value_name, column_mapping[value_name], style_df, palette_2[1]))
    elif language == 'danish':
        for value_name in style_df['Metric']:
            if value_name in column_mapping:
                content.append(value_boxes_12(value_name, column_mapping[value_name], style_df, palette_2[1]))
    
    content.append(metrics_explanation('Stylometrics', stylometrics_explanation_text, "collapse-button_1", "collapse_1"))

    return html.Div(children = content, style = {"backgroundColor": personal_palette[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})


def sent_func(sent_df: pd.DataFrame, sentiment_explanation_text: str, arcs: list, sentiment_method: str, language: str) -> html.Div:
    column_mapping = {
        'mean_sentiment': 'Mean Sentiment',
        'std_sentiment': 'Std Sentiment',
        'mean_sentiment_first_ten_percent': 'Mean Sentiment First 10%',
        'mean_sentiment_last_ten_percent': 'Mean Sentiment Last 10%',
        'difference_lastten_therest': 'Difference between last 10 and the rest',
        'hurst': 'Hurst',
        'approximate_entropy_value': 'Approximate Entropy'
    }

    if sentiment_method == 'syuzhet':
        # remove 'hurst': 'Hurst' from column_mapping and add 'HURST_SYUZHET': 'Hurst'
        column_mapping.pop('hurst')
        column_mapping['HURST_SYUZHET'] = 'Hurst'
    
    content = [html.H2(children='Sentiment', className="fw-bold text-white")]
    
    if sentiment_method == 'afinn':
        if language == 'english':
            for value_name in sent_df['Metric']:
                if value_name in column_mapping:
                    content.append(value_boxes_12(value_name, column_mapping[value_name], sent_df, palette_2[1]))
        elif language == 'danish':
            for value_name in sent_df['Metric']:
                if value_name in column_mapping:
                    content.append(value_boxes_12(value_name, column_mapping[value_name], sent_df, palette_2[1]))

    elif sentiment_method == 'vader':
        for value_name in sent_df['Metric']:
            if value_name in column_mapping:
                content.append(value_boxes_fig(value_name, column_mapping[value_name], sent_df, palette_2[1]))

    elif sentiment_method == 'syuzhet':
        for value_name in sent_df['Metric']:
            if value_name == 'HURST_SYUZHET':
                content.append(value_boxes_fig(value_name, column_mapping[value_name], sent_df, palette_2[1]))
            elif value_name in column_mapping:
                content.append(value_boxes_12(value_name, column_mapping[value_name], sent_df, palette_2[1]))
    
    elif sentiment_method == 'avg_syuzhet_vader':
        for value_name in sent_df['Metric']:
            if value_name in column_mapping:
                content.append(value_boxes_12(value_name, column_mapping[value_name], sent_df, palette_2[1]))

    if arcs is not None:
        content.append(value_boxes_arcs_fig('Progression of Sentiment Arcs', arcs))
    
    content.append(metrics_explanation('Sentiment', sentiment_explanation_text, "collapse-button_2", "collapse_2"))
    
    return html.Div(content, style={"backgroundColor": palette_2[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def read_func(read_df: pd.DataFrame, readability_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Readability', className="fw-bold text-white"),
        value_boxes_fig('flesch_grade', 'Flesch Grade', read_df, palette_4[1]),
        value_boxes_fig('flesch_ease', 'Flesch Ease', read_df, palette_4[1]),
        value_boxes_fig('smog', 'Smog', read_df, palette_4[1]),
        value_boxes_fig('ari', 'Ari', read_df, palette_4[1]),
        value_boxes_fig('dale_chall_new', 'Dale Chall New', read_df, palette_4[1]),
        metrics_explanation('Readability', readability_explanation_text, "collapse-button_4", "collapse_4"),
    ], style = {"backgroundColor": palette_4[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def roget_func(roget_df: pd.DataFrame, roget_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Roget', className="fw-bold text-white"),
        dbc.Row([
            value_boxes_1('Your Value', palette_5[1],{'size': 2, 'offset': 3}),
            value_boxes_1('Canonical Mean', palette_5[1],{'size': 2, 'offset': 1}),
            value_boxes_1('Bestseller Mean', palette_5[1],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 10, "marginBottom": 10}),
        value_boxes_1234('roget_n_tokens', 'Roget n Tokens', roget_df, palette_5[1]),
        value_boxes_1234('roget_n_tokens_filtered', 'Roget Filtered', roget_df, palette_5[1]),
        value_boxes_1234('roget_n_cats', 'Roget n Categories', roget_df, palette_5[1]),
        metrics_explanation('Roget', roget_explanation_text, "collapse-button_5", "collapse_5"),
    ], style = {"backgroundColor": palette_5[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})