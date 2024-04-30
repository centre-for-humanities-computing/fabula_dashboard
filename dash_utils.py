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

def value_boxes_2(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['Value'].values[0].round(2)}", style=style_value_value),
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)

def value_boxes_3(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['Mean_Bestsellers'].values[0].round(2)}", style=style_value_value) if not df[df['Metric'] == column_name]['Mean_Bestsellers'].isna().any() else html.Div("NA", style=style_value_value)
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)

def value_boxes_4(column_name: str, df: pd.DataFrame, color: str, location: dict, extra_style: dict = {}) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{df[df['Metric'] == column_name]['Mean_Canonicals'].values[0].round(2)}", style=style_value_value) if not df[df['Metric'] == column_name]['Mean_Canonicals'].isna().any() else html.Div("NA", style=style_value_value)
            ])
        ], style = {"backgroundColor": color, 'borderColor': 'black', 'width': '60%', "margin": "0px 0px 0px 2.5vw"}|extra_style)
    ], width = location)


def metrics_explanation(metric_group: str, explanation: str, id_but: str, id_col) -> dbc.Row:
    return dbc.Row([
         dbc.Col([
              html.I(className="bi bi-caret-right-fill", n_clicks = 0, id=id_but, style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
              html.H5("Description of metrics", style={"display": "inline"}),
        ], width = {'size': 3, 'offset': 1}),
        dbc.Collapse(
            dbc.Card([
                html.H3(children=f"Explanation of {metric_group} Metrics", style = {'textAlign': 'left'}),
                html.Div(dcc.Markdown(explanation))
            ], style = {'backgroundColor': palette_1[1], 'borderColor': 'black', 'padding': '10px'}), id=id_col, is_open=False),
        ])

def styl_func(style_df: pd.DataFrame, stylometrics_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Stylometrics', className="fw-bold text-white"),
        dbc.Row([
            value_boxes_1('Your Value', palette_1[2],{'size': 2, 'offset': 3}),
            value_boxes_1('Canonical Mean', palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_1('Bestseller Mean', palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes_1('Word Count', palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2('word_count', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3('word_count', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4('word_count', style_df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20}),
        dbc.Row([
            value_boxes_1('Word Length', palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2('average_wordlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3('average_wordlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4('average_wordlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20}),
        dbc.Row([
            value_boxes_1('MSTTR', palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2('msttr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3('msttr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4('msttr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20}),
        dbc.Row([
            value_boxes_1('Sentence Length', palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2('average_sentlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3('average_sentlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4('average_sentlen', style_df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20}),
        dbc.Row([
            value_boxes_1('bzipr', palette_1[2],{'size': 2, 'offset': 0}),
            value_boxes_2('bzipr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_3('bzipr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
            value_boxes_4('bzipr', style_df, palette_1[2],{'size': 2, 'offset': 1}),
        ], style={"marginTop": 20, "marginBottom": 20}),
        metrics_explanation('Stylometrics', stylometrics_explanation_text, "collapse-button_1", "collapse_1"),
    ], style = {"backgroundColor": personal_palette[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

# def styl_func(style_df: pd.DataFrame, stylometrics_explanation_text: str) -> html.Div:
#     return html.Div([
#         html.H2(children='Stylometrics', className="fw-bold text-white"),
#         dbc.Row([
#             value_boxes('word_count', 'Word Count', style_df, palette_1[2]),
#             value_boxes('average_wordlen', 'Word Length', style_df, palette_1[2]),
#         ], style={"marginTop": 10, "marginBottom": 10}),
#         dbc.Row([
#             value_boxes('msttr', 'MSTTR', style_df, palette_1[2]),
#             value_boxes('average_sentlen', 'Average Sentence Length', style_df, palette_1[2]),
#         ], style={"marginTop": 10, "marginBottom": 10}),
#         dbc.Row([
#             value_boxes('bzipr', 'bzipr', style_df, palette_1[2]),
#         ], style={"marginTop": 10, "marginBottom": 10}),
#         metrics_explanation('Stylometrics', stylometrics_explanation_text, "collapse-button_1", "collapse_1"),
#     ], style = {"backgroundColor": personal_palette[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def sent_func(sent_df: pd.DataFrame, sentiment_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Sentiment', className="fw-bold text-white"),
        dbc.Row([
            value_boxes('mean_sentiment', 'Mean Sentiment', sent_df, palette_2[1]),
            value_boxes('mean_sentiment_first_ten_percent', 'Mean Sentiment First 10%', sent_df, palette_2[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('mean_sentiment_last_ten_percent', 'Mean Sentiment Last 10%', sent_df, palette_2[1]),
            value_boxes('difference_lastten_therest', 'Difference between last 10 and the rest', sent_df, palette_2[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        metrics_explanation('Sentiment', sentiment_explanation_text, "collapse-button_2", "collapse_2"),
    ], style = {"backgroundColor": palette_2[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def entro_func(entropy_df: pd.DataFrame, entropy_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Entropy', className="fw-bold text-white"),
        dbc.Row([
            value_boxes('word_entropy', 'Word Entropy', entropy_df, palette_3[1]),
            value_boxes('bigram_entropy', 'Bigram Entropy', entropy_df, palette_3[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('approximate_entropy_value', 'Approximate Entropy', entropy_df, palette_3[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        metrics_explanation('Entropy', entropy_explanation_text, "collapse-button_3", "collapse_3"),
    ], style = {"backgroundColor": palette_3[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def read_func(read_df: pd.DataFrame, readability_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Readability', className="fw-bold text-white"),
        dbc.Row([
            value_boxes('flesch_grade', 'Flesch Grade', read_df, palette_4[1]),
            value_boxes('flesch_ease', 'Flesch Ease', read_df, palette_4[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('smog', 'Smog', read_df, palette_4[1]),
            value_boxes('ari', 'Ari', read_df, palette_4[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('dale_chall_new', 'Dale Chall New', read_df, palette_4[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        metrics_explanation('Readability', readability_explanation_text, "collapse-button_4", "collapse_4"),
    ], style = {"backgroundColor": palette_4[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

def roget_func(roget_df: pd.DataFrame, roget_explanation_text: str) -> html.Div:
    return html.Div([
        html.H2(children='Roget', className="fw-bold text-white"),
        dbc.Row([
            value_boxes('roget_n_tokens', 'Roget n Tokens', roget_df, palette_5[1]),
            value_boxes('roget_n_tokens_filtered', 'Roget Filtered', roget_df, palette_5[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('roget_n_cats', 'Roget n Categories', roget_df, palette_5[1]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        metrics_explanation('Roget', roget_explanation_text, "collapse-button_5", "collapse_5"),
    ], style = {"backgroundColor": palette_5[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})