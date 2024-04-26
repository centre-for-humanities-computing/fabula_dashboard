from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

import base64
import datetime
import io

import pandas as pd
from statistics import mean 
from statistics import stdev

from metrics_function import *

quick_mode = 0

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

def floating_table(df: pd.DataFrame) -> dbc.Col:
    # return dbc.Col([
    #     dbc.Card([
    #         dbc.CardBody([
    #             html.Table([
    #                 html.Thead([
    #                     html.Tr([html.Th(col) for col in df.columns])
    #                 ]),
    #                 html.Tbody([
    #                     html.Tr([
    #                         html.Td(df.iloc[i][col]) for col in df.columns
    #                     ]) for i in range(len(df))
    #                 ])
    #             ])
    #         ])
    #     ])
    # ], width = {'size': 6, 'offset': 3})
    return html.P("Float", style = {'fontSize': 50, 'textAlign': 'center', 'margin': '10px'})




def value_boxes(column_name: str, value_name: str, df: pd.DataFrame, color: str) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(f"{value_name}", style=style_value_text),
                html.Div(f"{df[df['Metric'] == column_name]['Value'].values[0].round(2)}", style=style_value_value),
                html.Div(f"(Global Mean {value_name} {df[df['Metric'] == column_name]['Mean'].values[0].round(2)})", style=style_value_global) if not df[df['Metric'] == column_name]['Mean'].isna().any() else None
            ])
        ] if column_name in df['Metric'].values else None, 
        style = {"backgroundColor": color, 'borderColor': 'black'})
    ], width = {'size': 3, 'offset': 2})

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
            value_boxes('word_count', 'Word Count', style_df, palette_1[2]),
            value_boxes('average_wordlen', 'Word Length', style_df, palette_1[2]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('msttr', 'MSTTR', style_df, palette_1[2]),
            value_boxes('average_sentlen', 'Average Sentence Length', style_df, palette_1[2]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        dbc.Row([
            value_boxes('bzipr', 'bzipr', style_df, palette_1[2]),
        ], style={"marginTop": 10, "marginBottom": 10}),
        metrics_explanation('Stylometrics', stylometrics_explanation_text, "collapse-button_1", "collapse_1"),
    ], style = {"backgroundColor": personal_palette[0], "padding": "10px", "borderRadius": "15px", "margin": "10px"})

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

style_value_text = {'fontSize': 30, 'textAlign': 'center'}
style_value_value = {"textAlign": "center", "fontSize": 30}
style_value_figure = {'display': 'inline-block'}
style_value_global = {'fontSize': 15, 'textAlign': 'center'}

personal_palette = ['#5D5EDB', '#D353C2', '#FF5F98', '#FF8D6F', '#FFC45A', '#F9F871']
# palette_1 = ['#5D5EDB', '#7E79FA', '#9D94FF', '#BDB1FF', '#DDCFFF']  
# palette_2 = ['#920086', '#D353C2']
# palette_3 = ['#B20059', '#FF5F98']
# palette_4 = ['#A9432C', '#FF8D6F']
# palette_5 = ['#9C6D00', '#FFC45A']

palette_chc = ["#084444","#40a49c", "#40bcb4", "#e8f4f4"]
personal_palette = ["#40a49c", "#40bcb4", "#e8f4f4"]
palette_1 = ["#40a49c", "#40bcb4", "#e8f4f4"]  
palette_2 = ["#40a49c", "#e8f4f4"]
palette_3 = ["#40a49c", "#e8f4f4"]
palette_4 = ["#40a49c", "#e8f4f4"]
palette_5 = ["#40a49c", "#e8f4f4"]


# read in explanation filex
with open('src/assets/texts/metrics_explanation.txt', 'r') as file:
            explanation_text = file.read()

# read in explanation filex
with open('src/assets/texts/stylometrics_explanations.txt', 'r') as file:
            stylometrics_explanation_text = file.read()
with open('src/assets/texts/sentiment_explanations.txt', 'r') as file:
            sentiment_explanation_text = file.read()
with open('src/assets/texts/entropy_explanations.txt', 'r') as file:
            entropy_explanation_text = file.read()
with open('src/assets/texts/readability_explanations.txt', 'r') as file:
            readability_explanation_text = file.read()
with open('src/assets/texts/roget_explanations.txt', 'r') as file:
            roget_explanation_text = file.read()
with open('src/assets/texts/home_page.txt', 'r') as file:
            home_page_text = file.read()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    # "margin-left": "420px",
    # "margin-left": "0px",
    # "padding-left": "0px",
    # "padding": "100px",
}

sidebar = html.Div([
    html.Br(),
    html.H1("Settings", className="text-center fw-bold fs-2"),
    html.Br(),
    html.H3(children='Choose Language', className="fw-bold"),
    dcc.Dropdown(options=[{'value': 'english', 'label': 'English'}, {'value': 'danish', 'label': 'Danish'}], id='lang-dropdown', placeholder="Select a language", searchable = False, style = {'color': 'black'}),
    html.Br(),
    html.H3(children='Choose Sentiment Analysis', style={'margin-top': '50px'}, className="fw-bold"),
    dcc.Dropdown(id='sent-dropdown', placeholder="Select sentiment analysis method", searchable = False, style = {'color': 'black'}),
    html.Br(),
    html.H3(children='Upload File', style={'margin-top': '50px'}, className="fw-bold"),
    dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files (.txt)')]), style={'width': '100%',
                                                                                                          'height': '120px', 
                                                                                                          'lineHeight': '120px',
                                                                                                          'borderWidth': '1px',
                                                                                                          'borderStyle': 'dashed',
                                                                                                          'borderRadius': '5px',
                                                                                                          'textAlign': 'center',
                                                                                                          'margin': '10px 20px 0px 0px'},multiple=True),
    html.Br(),
    html.H3(children='Write Text', style={'margin-top': '50px'}, className="fw-bold"),
    dcc.Textarea(id='textarea-example',value=None,style={'width': '100%', 
                                                         'height': '120px', 
                                                         'textAlign': 'left',
                                                         'margin': '0px 20px 10px 0px'}),
    html.Button('Submit', id='submit-val', n_clicks=0, className = 'button', style = {}),
    html.Div(children = [html.P("Made by Fabula-NET", style = {'position': 'fixed', 'bottom': '0', 'left': '20px','color': 'white', 'padding': '10px'}), 
                        html.A([DashIconify(icon="ion:logo-github", width = 50, color = 'white', style = {'position': 'fixed', 'bottom': '15px', 'left': '200px', 'cursor': 'pointer'})
                                ], href = "https://github.com/centre-for-humanities-computing/fabula_pipeline/", target = "_blank"),
                        html.A([DashIconify(icon="ph:globe", width = 55, color = 'white', style = {'position': 'fixed', 'bottom': '12px', 'left': '265px', 'cursor': 'pointer'})
                                ], href = "https://centre-for-humanities-computing.github.io/fabula-net/", target = "_blank")
                        ]),
], className="bg-dark text-white", style=SIDEBAR_STYLE)

main_content = html.Div(
    children = [
        dbc.Row([html.Hr(style = {'margin': '10px'}), dbc.Nav([dbc.NavLink("Home", href="/", active="exact"),], vertical=False, pills=True,), html.Hr(style = {'margin': '10px'}),], id = 'output-data-upload'),
        dcc.Loading(id = 'loading-1', type = 'cube', children = [dcc.Store(id='intermediate-value'), dcc.Store(id='intermediate-value-2', data = 'string'), dcc.Store(id='file_or_text', data = 'string'),], fullscreen = False, style = {'position': 'fixed', 'top': '50%', 'left': '62.5%', 'transform': 'translate(-50%, -50%)'}, color = 'green'),
    ],
    style = CONTENT_STYLE
)

page_content = html.Div(id='page-content', style = {'padding': '10px'})

# app.layout = html.Div([
#     html.Div([sidebar, main_content], className="row")
# ], className="container-fluid", style={"height": "100vh"})
app.layout = dbc.Container([
       dcc.Location(id="url"),
       dbc.Row([dbc.Col(sidebar),
             dbc.Col([
                  dbc.Row(main_content),
                  dbc.Row(page_content)
                  ], width = {'size': 9, 'offset': 3})])],
    className="container-fluid", style={}, fluid = True)

def parse_contents(contents, filename, date, language, sentiment, text, fileortext):
    

    if language is None:
        print(Exception)
        return html.Div([dbc.Row([html.Hr(style = {'margin': '10px'}),dbc.Nav([dbc.NavLink("Home", href="/", active="exact"),], vertical=False, pills=True,),html.Hr(style = {'margin': '10px'}),]),]), None, None
    
    if sentiment is None:
        print(Exception)
        return html.Div([dbc.Row([html.Hr(style = {'margin': '10px'}),dbc.Nav([dbc.NavLink("Home", href="/", active="exact"),], vertical=False, pills=True,),html.Hr(style = {'margin': '10px'}),]),]), None, None

    if fileortext == 'file':
        # if contents == ...:
        content_type, content_string = contents[0].split(',')
        print(content_string)

        decoded = base64.b64decode(content_string)
        try:
            if 'txt' in filename[0]:
                full_string = decoded.decode('utf-8')
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        
    if fileortext == 'text':
        full_string = text

    dict_0 = compute_metrics(full_string, language, sentiment)
    print("Done with computing metrics")

    # compute metrics is producing lists in its dict, and don't know what to do with them
    # so I'm just going to take the mean of all the lists and put them in the dict
    if language == 'english':
        if len(dict_0['concreteness']) > 0:
            dict_0['concreteness_mean'] = mean([i[0] for i in dict_0['concreteness']])
        if len(dict_0['concreteness']) > 1:
            dict_0['concreteness_sd'] = stdev([i[0] for i in dict_0['concreteness']])
        if len(dict_0['valence']) > 0:
            dict_0['valence_mean'] = mean([float(i[0]) for i in dict_0['valence']])
        if len(dict_0['valence']) > 1:
            dict_0['valence_sd'] = stdev([float(i[0]) for i in dict_0['valence']])
        if len(dict_0['arousal']) > 0:
            dict_0['arousal_mean'] = mean([float(i[0]) for i in dict_0['arousal']])
        if len(dict_0['arousal']) > 1:
            dict_0['arousal_sd'] = stdev([float(i[0]) for i in dict_0['arousal']])
        if len(dict_0['dominance']) > 0:
            dict_0['dominance_mean'] = mean([float(i[0][:-3]) for i in dict_0['dominance']])
        if len(dict_0['dominance']) > 1:
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
    # style_df = concat_df[concat_df['Metric'].isin(['word_count', 'average_wordlen', 'msttr', 'average_sentlen', 'bzipr'])]
    # sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest', 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]
    # entropy_df = concat_df[concat_df['Metric'].isin(['word_entropy', 'bigram_entropy', 'approximate_entropy_value'])]
    # read_df = concat_df[concat_df['Metric'].isin(['flesch_grade', 'flesch_ease', 'smog', 'ari', 'dale_chall_new'])]
    # roget_df = concat_df[concat_df['Metric'].isin(['roget_n_tokens', 'roget_n_tokens_filtered', 'roget_n_cats'])]

    if language == 'english':
         navbar = html.Div([
              dbc.Row([
                   html.Hr(style = {'margin': '10px'}),  # horizontal line
                   dbc.Nav([
                        dbc.NavLink("Home", href="/", active="exact"),
                        dbc.NavLink("Stylometrics", href="/styl", active="exact"),
                        dbc.NavLink("Sentiment", href="/sent", active="exact"),
                        dbc.NavLink("Entropy", href="/entro", active="exact"),
                        dbc.NavLink("Readability", href="/read", active="exact"),
                        dbc.NavLink("Roget", href="/roget", active="exact"),
                        dbc.NavLink("Float", href="/float", active="exact"),
                        ], vertical=False, pills=True,),
                    html.Hr(style = {'margin': '10px'}),  # horizontal line
                    ]),
                ])
    
    if language == 'danish':
         navbar = html.Div([
              dbc.Row([
                   html.Hr(style = {'margin': '10px'}),  # horizontal line
                   dbc.Nav([
                        dbc.NavLink("Home", href="/", active="exact"),
                        dbc.NavLink("Stylometrics", href="/styl", active="exact"),
                        dbc.NavLink("Sentiment", href="/sent", active="exact"),
                        dbc.NavLink("Entropy", href="/entro", active="exact"),
                        ], vertical=False, pills=True,),
                    html.Hr(style = {'margin': '10px'}),  # horizontal line
                    ]),
                ])

    return navbar, concat_df.to_dict(), full_string

def quick_parse():

    # read csv
    concat_df = pd.read_csv('data/output.csv')

    # use only specified rows from concat_df
    style_df = concat_df[concat_df['Metric'].isin(['word_count', 'average_wordlen', 'msttr', 'average_sentlen', 'bzipr'])]
    sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest', 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]
    entropy_df = concat_df[concat_df['Metric'].isin(['word_entropy', 'bigram_entropy', 'approximate_entropy_value'])]
    read_df = concat_df[concat_df['Metric'].isin(['flesch_grade', 'flesch_ease', 'smog', 'ari', 'dale_chall_new'])]
    roget_df = concat_df[concat_df['Metric'].isin(['roget_n_tokens', 'roget_n_tokens_filtered', 'roget_n_cats'])]

    return html.Div([
        
        dbc.Row([
            html.Hr(),  # horizontal line

            html.Hr(),  # horizontal line
        ]),

        dbc.Row([
             dbc.Nav([
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Stylometrics", href="/styl", active="exact"),
                dbc.NavLink("Sentiment", href="/sent", active="exact"),
                dbc.NavLink("Entropy", href="/entro", active="exact"),
                dbc.NavLink("Readabnility", href="/read", active="exact"),
                dbc.NavLink("Roget", href="/roget", active="exact"),
             ], vertical=False, pills=True)
        ]),

    html.Hr(),  # horizontal line
])

@app.callback(
    Output("lang-dropdown", "style"),
    State("lang-dropdown", "value"),
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True
)
def set_dropdown_required_lang(value, n_clicks):
    if n_clicks > 0:
        if value is None:
            return {"border": "2px solid red", 'color': 'black'}
        else:
            return {'color': 'black'}

@app.callback(
    Output("sent-dropdown", "style"),
    State("sent-dropdown", "value"),
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True
)
def set_dropdown_required_sent(value, n_clicks):
    if n_clicks > 0:
        if value is None:
            return {"border": "2px solid red", 'color': 'black'}
        else:
            return {'color': 'black'}

@app.callback(
    Output('sent-dropdown', "options"),
    Input('lang-dropdown', "value")
)
def update_options(value):
    if value=='english':
        return [{'label':'Afinn', 'value': 'afinn', 'title': 'A dictionary approach to sentiment analysis developed by Afinn'}, {'label':'Vader', 'value': 'vader'}, {'label':'Syuzhet', 'value': 'syuzhet'}, {'label':'Avg Syuzhet Vader', 'value': 'avg_syuzhet_vader'}]
    if value=='danish':
        return [{'label':'Afinn', 'value': 'afinn'}]
    else:
        raise PreventUpdate

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

@callback(Output('file_or_text', 'data'),
          Input('upload-data', 'filename'),
          Input('textarea-example', 'value'),)
def file_or_text(filename, text):
    if ctx.triggered_id == 'upload-data':
        return 'file'
    if ctx.triggered_id == 'textarea-example':
        return 'text'

if quick_mode == 1:
    @callback(Output('output-data-upload', 'children'),
              Output('intermediate-value', 'data'),
              Input('submit-val', 'n_clicks'),
              prevent_initial_call=True)
    def update_output(n_clicks):
        return quick_parse(), pd.read_csv('data/output.csv').to_dict()
else:
    @callback(Output('output-data-upload', 'children'),
              Output('intermediate-value', 'data'),
              Output('intermediate-value-2', 'data'),
                State('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'),
                State('lang-dropdown', 'value'),
                State('sent-dropdown', 'value'),
                State('textarea-example', 'value'),
                State('file_or_text', 'data'),
                Input('submit-val', 'n_clicks'),
                prevent_initial_call=True)
    def update_output(list_of_contents, list_of_names, list_of_dates, language, sentiment, text, fileortext, n_clicks):
        if language is None:
            raise PreventUpdate

        if sentiment is None:
            raise PreventUpdate
        
        if n_clicks > 0:
            if list_of_contents is not None or text is not None:
                children, data, text_string = parse_contents(list_of_contents, list_of_names, list_of_dates, language, sentiment, text, fileortext)

                return children, data, text_string
            
            # if text is not None:
            #     children = parse_contents(text, list_of_names[0], list_of_dates[0], language, sentiment, text)

            #     return children

# @callback(Output('output-data-upload', 'children'),
#             Output('intermediate-value', 'data'),
#             Output('intermediate-value-2', 'data'),
#             State('upload-data', 'contents'),
#             State('upload-data', 'filename'),
#             State('upload-data', 'last_modified'),
#             State('lang-dropdown', 'value'),
#             State('sent-dropdown', 'value'),
#             State('textarea-example', 'value'),
#             Input('submit-val', 'n_clicks'),
#             prevent_initial_call=True)
# def update_output(list_of_contents, list_of_names, list_of_dates, language, sentiment, text, n_clicks):
#     if n_clicks > 0:
#         print(ctx.inputs)
#         if list_of_contents is not None:
#             children, data, text_string = parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0], language, sentiment, text)

#             return children, data, text_string

@callback(Output("page-content", "children"),
          Input("url", "pathname"),
          Input("intermediate-value", "data"),
          Input('submit-val', 'n_clicks'),
          State('upload-data', 'contents'),
          State('file_or_text', 'data'),
          State('lang-dropdown', 'value'),
          State('sent-dropdown', 'value'),
          Input("intermediate-value-2", "data"),)
def render_page_content(pathname, data, n_clicks, contents, text, language, sentiment, full_string):
    if n_clicks > 0:
        if language is None:
            raise PreventUpdate

        if sentiment is None:
            raise PreventUpdate
        
        if contents is not None or text is not None:
            if language == 'english':
                concat_df = pd.DataFrame.from_dict(data)

                # use only specified rows from concat_df
                style_df = concat_df[concat_df['Metric'].isin(['word_count', 'average_wordlen', 'msttr', 'average_sentlen', 'bzipr'])]
                sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest', 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]
                entropy_df = concat_df[concat_df['Metric'].isin(['word_entropy', 'bigram_entropy', 'approximate_entropy_value'])]
                read_df = concat_df[concat_df['Metric'].isin(['flesch_grade', 'flesch_ease', 'smog', 'ari', 'dale_chall_new'])]
                roget_df = concat_df[concat_df['Metric'].isin(['roget_n_tokens', 'roget_n_tokens_filtered', 'roget_n_cats'])]

                if pathname == "/":
                    return html.Div([
                        html.P("Welcome to Fabula-NET", style = {'fontSize': 50, 'textAlign': 'center', 'margin': '10px'}),
                        html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'center', 'margin': '10px'}),])
                elif pathname == "/styl":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        styl_func(style_df=style_df, stylometrics_explanation_text=stylometrics_explanation_text)])
                elif pathname == "/sent":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        sent_func(sent_df=sent_df, sentiment_explanation_text=sentiment_explanation_text)])
                elif pathname == "/entro":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        entro_func(entropy_df=entropy_df, entropy_explanation_text=entropy_explanation_text)])
                elif pathname == "/read":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        read_func(read_df=read_df, readability_explanation_text=readability_explanation_text)])
                elif pathname == "/roget":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        roget_func(roget_df=roget_df, roget_explanation_text=roget_explanation_text)])
                elif pathname == "/float":
                     return html.Div([floating_table(concat_df)])
            if language == 'danish':
                concat_df = pd.DataFrame.from_dict(data)

                # use only specified rows from concat_df
                style_df = concat_df[concat_df['Metric'].isin(['word_count', 'average_wordlen', 'msttr', 'average_sentlen', 'bzipr'])]
                sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest', 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]
                entropy_df = concat_df[concat_df['Metric'].isin(['word_entropy', 'bigram_entropy', 'approximate_entropy_value'])]

                if pathname == "/":
                    return html.Div([
                        html.P("Welcome to Fabula-NET", style = {'fontSize': 50, 'textAlign': 'center', 'margin': '10px'}),
                        html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'center', 'margin': '10px'}),])
                elif pathname == "/styl":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        styl_func(style_df=style_df, stylometrics_explanation_text=stylometrics_explanation_text)])
                elif pathname == "/sent":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        sent_func(sent_df=sent_df, sentiment_explanation_text=sentiment_explanation_text)])
                elif pathname == "/entro":
                    return html.Div([
                        dbc.Row([html.P(children=['First 500 characters:'], className="fw-bold fs-10"),html.P(children=[full_string[:500], '...']), html.Hr()]),
                        entro_func(entropy_df=entropy_df, entropy_explanation_text=entropy_explanation_text)])
        # If the user tries to reach a different page, return a 404 message
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            className="p-3 bg-light rounded-3",
        )
    elif pathname == "/":
        return html.Div([
            html.P("Welcome to Fabula-NET", style = {'fontSize': 50, 'textAlign': 'center', 'margin': '10px'}),
            html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'center', 'margin': '10px'}),])

# @callback(Output('output-data-upload', 'children'),
#           Input('intermediate-value', 'data'))
# def return_func(data):
#     return data
            

if __name__ == '__main__':
    app.run(debug=True)