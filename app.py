from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

import base64
import os

import pandas as pd
from statistics import mean 
from statistics import stdev

from metrics_function import *
from dash_utils import *
from fabulanet_func import *

load_figure_template("minty")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read in explanation filex
with open(os.path.join('assets', 'texts', 'metrics_explanation.txt'), 'r') as file:
            explanation_text = file.read()

# read in explanation filex
with open(os.path.join('assets', 'texts', 'stylometrics_explanations.txt'), 'r') as file:
            stylometrics_explanation_text = file.read()
with open(os.path.join('assets', 'texts', 'sentiment_explanations.txt'), 'r') as file:
            sentiment_explanation_text = file.read()
with open(os.path.join('assets', 'texts', 'entropy_explanations.txt'), 'r') as file:
            entropy_explanation_text = file.read()
with open(os.path.join('assets', 'texts', 'readability_explanations.txt'), 'r') as file:
            readability_explanation_text = file.read()
with open(os.path.join('assets', 'texts', 'roget_explanations.txt'), 'r') as file:
            roget_explanation_text = file.read()
with open(os.path.join('assets', 'texts', 'home_page.txt'), 'r') as file:
            home_page_text = file.read()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],suppress_callback_exceptions=True)
server = app.server

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
    html.H3(children='Choose Group (not working)', style={'margin-top': '50px'}, className="fw-bold"),
    dcc.Dropdown(options=[{'value': 'canonical', 'label': 'Canonical'}, {'value': 'bestseller', 'label': 'Bestseller'}], id='group-dropdown', placeholder="Select a Group", searchable = False, style = {'color': 'black'}, multi = True),
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
)

page_content = html.Div(id='page-content', style = {'padding': '10px'})

app.layout = dbc.Container([
       dcc.Location(id="url"),
       dbc.Row([dbc.Col(sidebar),
             dbc.Col([
                  dbc.Row(main_content),
                  dbc.Row(page_content)
                  ], width = {'size': 9, 'offset': 3})])],
    className="container-fluid", style={}, fluid = True)

@callback(
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

@callback(
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

@callback(
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

@callback(
    Output("collapse_1", "is_open"),
    [Input("collapse-button_1", "n_clicks")],
    [State("collapse_1", "is_open")],
)
def toggle_collapse_1(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("collapse_2", "is_open"),
    [Input("collapse-button_2", "n_clicks")],
    [State("collapse_2", "is_open")],
)
def toggle_collapse_2(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("collapse_3", "is_open"),
    [Input("collapse-button_3", "n_clicks")],
    [State("collapse_3", "is_open")],
)
def toggle_collapse_3(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output("collapse_4", "is_open"),
    [Input("collapse-button_4", "n_clicks")],
    [State("collapse_4", "is_open")],
)
def toggle_collapse_4(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
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

@callback(Output('output-data-upload', 'children'),
            Output('intermediate-value', 'data'),
            Output('intermediate-value-2', 'data'),
            State('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('lang-dropdown', 'value'),
            State('sent-dropdown', 'value'),
            State('textarea-example', 'value'),
            State('file_or_text', 'data'),
            Input('submit-val', 'n_clicks'),
            prevent_initial_call=True)
def update_output(list_of_contents, list_of_names, language, sentiment, text, fileortext, n_clicks):
    if language is None:
        raise PreventUpdate

    if sentiment is None:
        raise PreventUpdate
    
    if n_clicks > 0:
        if list_of_contents is not None or text is not None:
            children, data, text_string = parse_contents(list_of_contents, list_of_names, language, sentiment, text, fileortext)

            return children, data, text_string


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
                        html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'left', 'margin': '10px'}),])
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
                elif pathname == "/about":
                     return html.Div(["Add further information about metrics and stuff here"])
            if language == 'danish':
                concat_df = pd.DataFrame.from_dict(data)

                # use only specified rows from concat_df
                style_df = concat_df[concat_df['Metric'].isin(['word_count', 'average_wordlen', 'msttr', 'average_sentlen', 'bzipr'])]
                sent_df = concat_df[concat_df['Metric'].isin(['mean_sentiment', 'std_sentiment', 'mean_sentiment_first_ten_percent', 'mean_sentiment_last_ten_percent', 'difference_lastten_therest', 'arc_mean', 'arc_sd', 'mean_sentiment_per_segment_mean', 'mean_sentiment_per_segment_sd'])]
                entropy_df = concat_df[concat_df['Metric'].isin(['word_entropy', 'bigram_entropy', 'approximate_entropy_value'])]

                if pathname == "/":
                    return html.Div([
                        html.P("Welcome to Fabula-NET", style = {'fontSize': 50, 'textAlign': 'center', 'margin': '10px'}),
                        html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'left', 'margin': '10px'}),])
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
                elif pathname == "/about":
                     return html.Div(["Add further information about metrics and stuff here"])
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
            html.P(dcc.Markdown(home_page_text), style = {'fontSize': 20, 'textAlign': 'left', 'margin': '10px'}),])

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True)