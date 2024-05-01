from dash import html
import dash_bootstrap_components as dbc
import base64

import pandas as pd
from statistics import mean 
from statistics import stdev

from metrics_function import *
from dash_utils import *

def parse_contents(contents, filename, language, sentiment, text, fileortext):

    if language is None:
        print(Exception)
        return html.Div([dbc.Row([html.Hr(style = {'margin': '10px'}),dbc.Nav([dbc.NavLink("Home", href="/", active="exact"),], vertical=False, pills=True,),html.Hr(style = {'margin': '10px'}),]),]), None, None
    
    if sentiment is None:
        print(Exception)
        return html.Div([dbc.Row([html.Hr(style = {'margin': '10px'}),dbc.Nav([dbc.NavLink("Home", href="/", active="exact"),], vertical=False, pills=True,),html.Hr(style = {'margin': '10px'}),]),]), None, None

    ### removing syuzhet for render deployment
    if sentiment == 'syuzhet':
        sentiment = 'afinn'
    if sentiment == 'avg_syuzhet_vader':
        sentiment = 'afinn'
    ###

    if fileortext == 'file':
        # if contents == ...:
        content_type, content_string = contents[0].split(',')

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

    # read in mean file
    mean_df = pd.read_csv(os.path.join('data', 'mean.csv'))


    column_names_row = pd.DataFrame([df.columns], columns=df.columns)
    column_names_row = column_names_row.T
    common_columns = df.columns.intersection(mean_df.columns)
    mean_df_common = mean_df[common_columns].T
    df = df.T
    concat_df = pd.concat([column_names_row, df, mean_df_common], ignore_index=True, axis = 1)
    concat_df.columns = ['Metric', 'Value', 'Mean_Bestsellers', 'Mean_Canonicals']

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
                        dbc.NavLink("About", href="/about", active="exact"),
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