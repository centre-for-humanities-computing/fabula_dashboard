from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

import base64
import datetime
import io

import pandas as pd
from statistics import mean 
from statistics import stdev

from metrics_function import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read in mean file
mean_df = pd.read_csv('data/mean.csv')

# read in explanation file
with open('data/metrics_explanation.txt', 'r') as file:
            explanation_text = file.read()

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children = [
        html.H1(children='Language in Text'),
        dcc.Dropdown(options={'english': 'English', 'danish': 'Danish'}, id='lang-dropdown', placeholder="Select a lagnguage"),
        html.H1(children='Sentiment Analysis to use'),
        dcc.Dropdown(['afinn', 'vader', 'syuzhet', 'avg_syuzhet_vader'], id='sent-dropdown', placeholder="Select sentiment analysis method"),
        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), style={'width': '100%','height': '120px', 'lineHeight': '120px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},multiple=True),
        dcc.Textarea(id='textarea-example',value=None,style={'width': '100%', 'height': '120px', 'textAlign': 'center','margin': '10px'}),
        html.Button('Submit', id='submit-val', n_clicks=0),
        html.Div(id='output-data-upload'),
        html.H2(children='Explanation of Metrics'),
        dcc.Markdown(explanation_text),
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
    dict_0['concreteness_mean'] = mean([i[0] for i in dict_0['concreteness']])
    dict_0['concreteness_sd'] = stdev([i[0] for i in dict_0['concreteness']])
    dict_0['valence_mean'] = mean([float(i[0]) for i in dict_0['valence']])
    dict_0['valence_sd'] = stdev([float(i[0]) for i in dict_0['valence']])
    dict_0['arousal_mean'] = mean([float(i[0]) for i in dict_0['arousal']])
    dict_0['arousal_sd'] = stdev([float(i[0]) for i in dict_0['arousal']])
    dict_0['dominance_mean'] = mean([float(i[0][:-3]) for i in dict_0['dominance']])
    dict_0['dominance_sd'] = stdev([float(i[0][:-3]) for i in dict_0['dominance']])
    dict_0['arc_mean'] = mean(dict_0['arc'])
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

    ])

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              Input('lang-dropdown', 'value'),
              Input('sent-dropdown', 'value'),
              Input('textarea-example', 'value'),
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

if __name__ == '__main__':
    app.run(debug=True)