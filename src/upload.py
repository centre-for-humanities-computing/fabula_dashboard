from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

import base64
import datetime
import io

import pandas as pd
from statistics import mean 
from statistics import stdev

from metrics_function import *

load_figure_template("minty")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read in mean file
mean_df = pd.read_csv('data/mean.csv')

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

style_value_text = {'fontSize': 50, 'textAlign': 'center'}
style_value_value = {"textAlign": "center", "fontSize": 30}
style_value_figure = {'display': 'inline-block'}
style_value_global = {'fontSize': 15, 'textAlign': 'center'}

personal_palette = ['#5D5EDB', '#D353C2', '#FF5F98', '#FF8D6F', '#FFC45A', '#F9F871']

# read in explanation filex
with open('data/metrics_explanation.txt', 'r') as file:
            explanation_text = file.read()

# read in explanation filex
with open('data/sentiment_explanations.txt', 'r') as file:
            sentiment_text = file.read()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)

app.layout = html.Div(
    children = [
        html.H1(children='Language in Text'),
        dcc.Dropdown(options={'english': 'English', 'danish': 'Danish'}, id='lang-dropdown', placeholder="Select a lagnguage"),
        html.H1(children='Sentiment Analysis to use'),
        dcc.Dropdown(['afinn', 'vader', 'syuzhet', 'avg_syuzhet_vader'], id='sent-dropdown', placeholder="Select sentiment analysis method"),
        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), style={'width': '100%','height': '120px', 'lineHeight': '120px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},multiple=True),
        dcc.Textarea(id='textarea-example',value=None,style={'width': '100%', 'height': '120px', 'textAlign': 'center','margin': '10px'}),
        html.Button('Submit', id='submit-val', n_clicks=0, className = 'button'),
        # html.Div(id='spinner-status'),
        dcc.Loading(id = 'loading-1', type = 'cube', children = [html.Div(id='output-data-upload')], fullscreen = False),
        #html.Div(id='output-data-upload'),
        html.H2(children='Explanation of Metrics'),
        dcc.Markdown(explanation_text),
        # dcc.Store stores the intermediate value
        dcc.Store(id='intermediate-value'),
    ]
)
 

def parse_contents(contents, filename, date, language, sentiment, text):
    

    if language is None:
        print(Exception)
        return html.Div(['Choose langauge'], style = {'color': 'red', 'fontSize': 50, 'textAlign': 'center', 'margin': '10px'})
    
    if sentiment is None:
        print(Exception)
        return html.Div(['Choose sentiment'], style = {'color': 'red', 'fontSize': 50, 'textAlign': 'center', 'margin': '10px'})

    if filename is not None:
        # if contents == ...:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'txt' in filename:
                full_string = decoded.decode('utf-8')
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
    if filename is None:
        full_string = contents

    dict_0 = compute_metrics(full_string, language, sentiment)
    print("Done with computing metrics")

    # compute metrics is producing lists in its dict, and don't know what to do with them
    # so I'm just going to take the mean of all the lists and put them in the dict
    if language == 'english':
        dict_0['concreteness_mean'] = mean([i[0] for i in dict_0['concreteness']])
        dict_0['concreteness_sd'] = stdev([i[0] for i in dict_0['concreteness']])
        dict_0['valence_mean'] = mean([float(i[0]) for i in dict_0['valence']])
        dict_0['valence_sd'] = stdev([float(i[0]) for i in dict_0['valence']])
        dict_0['arousal_mean'] = mean([float(i[0]) for i in dict_0['arousal']])
        dict_0['arousal_sd'] = stdev([float(i[0]) for i in dict_0['arousal']])
        dict_0['dominance_mean'] = mean([float(i[0][:-3]) for i in dict_0['dominance']])
        dict_0['dominance_sd'] = stdev([float(i[0][:-3]) for i in dict_0['dominance']])
    
    if 'arc' in dict_0:
        if len(dict_0['arc']) > 0:
            dict_0['arc_mean'] = mean(dict_0['arc'])
        if len(dict_0['arc']) > 1:
            dict_0['arc_sd'] = stdev(dict_0['arc'])
    if 'mean_sentiment_per_segment' in dict_0:
        dict_0['mean_sentiment_per_segment_mean'] = mean(dict_0['mean_sentiment_per_segment'])
        dict_0['mean_sentiment_per_segment_sd'] = stdev(dict_0['mean_sentiment_per_segment'])
    if 'approximate_entropy' in dict_0:
        dict_0['approximate_entropy_value'] = dict_0['approximate_entropy'][0]

    print(dict_0['approximate_entropy'])

    dict_1 = {k: [v] for k, v in dict_0.items() if k not in ['concreteness', 'valence', 'arousal', 'dominance', 'arc', 'mean_sentiment_per_segment', 'approximate_entropy']}

    df = pd.DataFrame(dict_1)
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype(float)

    column_names_row = pd.DataFrame([df.columns], columns=df.columns)
    column_names_row = column_names_row.T
    common_columns = df.columns.intersection(mean_df.columns)
    mean_df_common = mean_df[common_columns].T
    df = df.T
    concat_df = pd.concat([column_names_row, df, mean_df_common], ignore_index=True, axis = 1)
    concat_df.columns = ['Metric', 'Value', 'Mean']

    # use only specified rows from concat_df
    sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest' 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]

    return html.Div([

        html.Hr(),  # horizontal line

        html.P(children=(full_string[:500])),

        html.Hr(),  # horizontal line

        html.P(children=(text)),

        html.Hr(),  # horizontal line

        dash_table.DataTable(
            concat_df.to_dict('records'),
            [{'name': i, 'id': i} for i in concat_df.columns],
            style_cell={'textAlign': 'left'},
            style_data={
                 'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                 {
                      'if': {'row_index': 'odd'},
                      'backgroundColor': 'rgb(220, 220, 220)',
                      }
                      ],
            style_header={
                 'backgroundColor': 'rgb(210, 210, 210)',
                 'color': 'black',
                 'fontWeight': 'bold'
                 }
        ),

        html.Hr(),  # horizontal line

    html.Div([
        html.H2(children='Stylometrics'),
        dbc.Container([
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Table of Sentiment Metrics", style = {'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold', 'color': 'black'}),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=sent_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in sent_df.columns],
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold',
                                'width': 'auto'
                            }
                        )
                    )
                ],
                style={"marginTop": 10, "marginBottom": 10, 'width': '80%', 'float': 'left'}),
            ], style = {}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            dcc.Graph(
                                figure=go.Figure(
                                    data=[go.Bar(x=['Text', 'Mean'], y=[sent_df[sent_df['Metric'] == 'mean_sentiment']['Value'].values[0], sent_df[sent_df['Metric'] == 'mean_sentiment']['Mean'].values[0]], name='Sentiment')],
                                    layout=go.Layout(title='Mean Sentiment')
                                )
                            )
                        )
                    ],
                    style={"marginTop": 20, "marginBottom": 20}),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(
                            dcc.Graph(
                                figure=go.Figure(
                                    data=[go.Bar(x=['Text', 'Mean'], y=[sent_df[sent_df['Metric'] == 'std_sentiment']['Value'].values[0], sent_df[sent_df['Metric'] == 'std_sentiment']['Mean'].values[0]], name='Sentiment')],
                                    layout=go.Layout(title='Standard Deviation of Sentiment')
                                )
                            )
                        )
                    ],
                    style={"marginTop": 20, "marginBottom": 20}),
                ]),
            ]) if 'mean_sentiment' in sent_df['Metric'].values else None,
            dbc.Row([
                html.Div([
                html.I(className="bi bi-caret-right-fill", n_clicks = 0, id="collapse-button_2", style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
                html.H5("Description of metrics", style={"display": "inline"}),
                ]),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Explanation of Sentiment Metrics"),
                        dbc.CardBody(
                            dcc.Markdown("hello")
                        )
                    ]),
                    id="collapse_2",
                    is_open=False,
                ),
            ]),
        ])
    ], style={"backgroundColor": personal_palette[0], "padding": "10px", "borderRadius": "15px",  "display": "inline-block", "width": "100%", 'margin': '10px'},),

    html.Div([
        html.H2(children='Sentiment'),
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div(
                        [
                            html.Div(f"Mean Sentiment", style=style_value_text),
                            html.Div(
                                id=f"value-1",
                                children=f"{sent_df[sent_df['Metric'] == 'mean_sentiment']['Value'].values[0].round(2)}",
                                style=style_value_value,
                            ),
                            html.Div(
                                f"(Global Mean Sentiment {sent_df[sent_df['Metric'] == 'mean_sentiment']['Mean'].values[0].round(2)})", 
                                style=style_value_global),
                        ],
                        style={'backgroundColor': '#5BBB82', 'borderRadius': '15px', 'padding': '10px', 'display': 'inline-block','border': '1px solid black'},
                    ) if 'mean_sentiment' in sent_df['Metric'].values else None
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Div(f"Mean Sentiment First 10%", style=style_value_text),
                            html.Div(
                                id=f"value-1",
                                children=f"{sent_df[sent_df['Metric'] == 'mean_sentiment_first_ten_percent']['Value'].values[0].round(2)}",
                                style=style_value_value,
                            ),
                            html.Div(
                                f"(Global Mean Sentiment {sent_df[sent_df['Metric'] == 'mean_sentiment_first_ten_percent']['Mean'].values[0].round(2)})", 
                                style=style_value_global),
                        ],
                        style={'display': 'inline-block'},
                    ) if 'mean_sentiment_first_ten_percent' in sent_df['Metric'].values else None
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Div(f"Mean Sentiment Last 10%", style=style_value_text),
                            html.Div(
                                id=f"value-1",
                                children=f"{sent_df[sent_df['Metric'] == 'mean_sentiment_last_ten_percent']['Value'].values[0].round(2)}",
                                style=style_value_value,
                            ),
                            html.Div(
                                f"(Global Mean Sentiment {sent_df[sent_df['Metric'] == 'mean_sentiment_last_ten_percent']['Mean'].values[0].round(2)})", 
                                style=style_value_global),
                        ],
                        style={'display': 'inline-block'},
                    ) if 'mean_sentiment_last_ten_percent' in sent_df['Metric'].values else None
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Div(f"Difference between last 10% and the rest", style=style_value_text),
                            html.Div(
                                id=f"value-1",
                                children=f"{sent_df[sent_df['Metric'] == 'difference_lastten_therest']['Value'].values[0].round(2)}",
                                style=style_value_value,
                            ),
                        ],
                        style={'display': 'inline-block'},
                    ) if 'difference_lastten_therest' in sent_df['Metric'].values else None
                ),
            ]),
            
            
            # dbc.Row([
            #     dbc.Card([
            #         dbc.CardHeader("Table of Sentiment Metrics", style = {'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold', 'color': 'black'}),
            #         dbc.CardBody(
            #             dash_table.DataTable(
            #                 data=sent_df.to_dict('records'),
            #                 columns=[{'name': i, 'id': i} for i in sent_df.columns],
            #                 style_cell={'textAlign': 'left'},
            #                 style_header={
            #                     'backgroundColor': 'white',
            #                     'fontWeight': 'bold',
            #                     'width': 'auto'
            #                 }
            #             )
            #         )
            #     ],
            #     style={"marginTop": 10, "marginBottom": 10, 'width': '80%', 'float': 'left'}),
            # ], style = {}),
            # dbc.Row([
            #     dbc.Col([
            #         dbc.Card([
            #             dbc.CardBody(
            #                 dcc.Graph(
            #                     figure=go.Figure(
            #                         data=[go.Bar(x=['Text', 'Mean'], y=[sent_df[sent_df['Metric'] == 'mean_sentiment']['Value'].values[0], sent_df[sent_df['Metric'] == 'mean_sentiment']['Mean'].values[0]], name='Sentiment')],
            #                         layout=go.Layout(title='Mean Sentiment')
            #                     )
            #                 )
            #             )
            #         ],
            #         style={"marginTop": 20, "marginBottom": 20}),
            #     ]),
            #     dbc.Col([
            #         dbc.Card([
            #             dbc.CardBody(
            #                 dcc.Graph(
            #                     figure=go.Figure(
            #                         data=[go.Bar(x=['Text', 'Mean'], y=[sent_df[sent_df['Metric'] == 'std_sentiment']['Value'].values[0], sent_df[sent_df['Metric'] == 'std_sentiment']['Mean'].values[0]], name='Sentiment')],
            #                         layout=go.Layout(title='Standard Deviation of Sentiment')
            #                     )
            #                 )
            #             )
            #         ],
            #         style={"marginTop": 20, "marginBottom": 20}),
            #     ]),
            # ]) if 'mean_sentiment' in sent_df['Metric'].values else None,
            dbc.Row([dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=list(range(21)), y=dict_0['arc'], mode='lines')],
                    layout=go.Layout(title='Arc Time Series', template = 'minty')
                )
            )]),
            dbc.Row([
                html.Div([
                html.I(className="bi bi-caret-right-fill", n_clicks = 0, id="collapse-button_1", style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
                html.H5("Description of metrics", style={"display": "inline"}),
                ]),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Explanation of Sentiment Metrics"),
                        dbc.CardBody(
                            dcc.Markdown(sentiment_text)
                        )
                    ]),
                    id="collapse_1",
                    is_open=False,
                ),
            ]),
        ])
    ], style={"backgroundColor": personal_palette[1], 
              "padding": "10px", 
              "borderRadius": "15px", 
                "display": "inline-block", 
                "width": "100%", 
                'margin': '10px'
                },),


    html.Div([
        html.H2(children='Entropy'),
        dbc.Container([
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Table of Sentiment Metrics", style = {'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold', 'color': 'black'}),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=sent_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in sent_df.columns],
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold',
                                'width': 'auto'
                            }
                        )
                    )
                ],
                style={"marginTop": 10, "marginBottom": 10, 'width': '80%', 'float': 'left'}),
            ], style = {}),
            dbc.Row([
                html.Div([
                html.I(className="bi bi-caret-right-fill", n_clicks = 0, id="collapse-button_3", style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
                html.H5("Description of metrics", style={"display": "inline"}),
                ]),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Explanation of Sentiment Metrics"),
                        dbc.CardBody(
                            dcc.Markdown("hello")
                        )
                    ]),
                    id="collapse_3",
                    is_open=False,
                ),
            ]),
        ])
    ], style={"backgroundColor": personal_palette[2], "padding": "10px", "borderRadius": "15px",  "display": "inline-block", "width": "100%", 'margin': '10px'},),


    html.Div([
        html.H2(children='Readability'),
        dbc.Container([
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Table of Sentiment Metrics", style = {'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold', 'color': 'black'}),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=sent_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in sent_df.columns],
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold',
                                'width': 'auto'
                            }
                        )
                    )
                ],
                style={"marginTop": 10, "marginBottom": 10, 'width': '80%', 'float': 'left'}),
            ], style = {}),
            dbc.Row([
                html.Div([
                html.I(className="bi bi-caret-right-fill", n_clicks = 0, id="collapse-button_4", style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
                html.H5("Description of metrics", style={"display": "inline"}),
                ]),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Explanation of Sentiment Metrics"),
                        dbc.CardBody(
                            dcc.Markdown("hello")
                        )
                    ]),
                    id="collapse_4",
                    is_open=False,
                ),
            ]),
        ])
    ], style={"backgroundColor": personal_palette[3], "padding": "10px", "borderRadius": "15px",  "display": "inline-block", "width": "100%", 'margin': '10px'},) if language == 'english' else None,


    html.Div([
        html.H2(children='Roget'),
        dbc.Container([
            dbc.Row([
                dbc.Card([
                    dbc.CardHeader("Table of Sentiment Metrics", style = {'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold', 'color': 'black'}),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=sent_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in sent_df.columns],
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold',
                                'width': 'auto'
                            }
                        )
                    )
                ],
                style={"marginTop": 10, "marginBottom": 10, 'width': '80%', 'float': 'left'}),
            ], style = {}),
            dbc.Row([
                html.Div([
                html.I(className="bi bi-caret-right-fill", n_clicks = 0, id="collapse-button_5", style={"fontSize": "30px", "color": "white", "cursor": "pointer"}),
                html.H5("Description of metrics", style={"display": "inline"}),
                ]),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Explanation of Sentiment Metrics"),
                        dbc.CardBody(
                            dcc.Markdown("hello")
                        )
                    ]),
                    id="collapse_5",
                    is_open=False,
                ),
            ]),
        ])
    ], style={"backgroundColor": personal_palette[4], "padding": "10px", "borderRadius": "15px",  "display": "inline-block", "width": "100%", 'margin': '10px'},) if language == 'english' else None,


])

@app.callback(
    Output("collapse_1", "is_open"),
    [Input("collapse-button_1", "n_clicks")],
    [State("collapse_1", "is_open")],
)
def toggle_collapse_1(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_2", "is_open"),
    [Input("collapse-button_2", "n_clicks")],
    [State("collapse_2", "is_open")],
)
def toggle_collapse_2(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_3", "is_open"),
    [Input("collapse-button_3", "n_clicks")],
    [State("collapse_3", "is_open")],
)
def toggle_collapse_3(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_4", "is_open"),
    [Input("collapse-button_4", "n_clicks")],
    [State("collapse_4", "is_open")],
)
def toggle_collapse_4(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_5", "is_open"),
    [Input("collapse-button_5", "n_clicks")],
    [State("collapse_5", "is_open")],
)
def toggle_collapse_5(n, is_open):
    if n:
        return not is_open
    return is_open

# @callback(Output('spinner-status', 'children'),
#           Input('submit-val', 'n_clicks'),
#           Input('intermediate-value', 'data'))
# def spinner_func(clicks, data = None):
#     if clicks > 0:
#         if data != None:
#             return print("hoops")
#         else:
#             return html.Div([html.P("Retrieving info about 1000s of papers, please give it a few seconds",
#                                             style = {'order': '1', 'font-size': '1.5rem', 'color':'rgba(3, 3, 3, 0.2)',
#                                                     'text-align': 'center', 'margin-top': '10vh'}),
#                                     #html.Img(src='assets/spinner.gif', style= {'order':'2', 'margin': 'auto'})
#                                     dbc.Spinner(size="sm"),
#                                     ],
#                                     style= {'display': 'flex', 'flex-direction':'column', 'justify-content': 'center',
#                                             'align-items': 'center', 'min-height': '400px', 'width':'60vw', 'margin': 'auto'})


@callback(Output('output-data-upload', 'children'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('lang-dropdown', 'value'),
              State('sent-dropdown', 'value'),
              State('textarea-example', 'value'),
              Input('submit-val', 'n_clicks'),
              prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, list_of_dates, language, sentiment, text, n_clicks):
    if n_clicks > 0:
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d, language, sentiment, text) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]

            return children
        
        if text is not None:
            children = parse_contents(text, list_of_names, list_of_dates, language, sentiment, text)

            return children

# @callback(Output('output-data-upload', 'children'),
#           Input('intermediate-value', 'data'))
# def return_func(data):
#     return data
            

if __name__ == '__main__':
    app.run(debug=True)