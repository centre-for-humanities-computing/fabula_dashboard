from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import datetime
import io

import pandas as pd

from metrics_function import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# read in mean file
mean_df = pd.read_csv('data/mean.csv')

# read in explanation file
#explanation_text = open('data/metrics_explanation.txt','r')
explanation_text = "this is a text"

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children = [
        html.H1(children='Language in Text'),
        dcc.Dropdown(options={'english': 'English', 'danish': 'Danish'}, id='lang-dropdown', placeholder="Select a lagnguage"),
        html.H1(children='Sentiment Analysis to use'),
        dcc.Dropdown(['afinn', 'vader', 'syuzhet', 'avg_syuzhet_vader'], id='sent-dropdown', placeholder="Select sentiment analysis method"),
        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), style={'width': '100%','height': '120px', 'lineHeight': '120px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},multiple=True),
        dcc.Textarea(id='textarea-example',value='Textarea content initialized\nwith multiple lines of text',style={'width': '100%', 'height': 100}),
        html.Div(id='output-data-upload'),
        html.H2(children='Explanation of Metrics'),
        html.P(children=explanation_text),
    ]
)

# app.layout = html.Div([
#     dcc.Dropdown(['NYC', 'MTL', 'SF'], 'NYC', id='demo-dropdown'),
#     html.Div(id='dd-output-container')
# ])

def parse_contents(contents, filename, date, language, sentiment, text):
    
    # if contents == ...:
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'txt' in filename:
            # Assume that the user uploaded a CSV file
            full_string = decoded.decode('utf-8')
        # elif 'xls' in filename:
        #     # Assume that the user uploaded an excel file
        #     df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
        
    # elif text == ...:
    #     full_string == text

    # else:
    #     print(e)
    #     return html.Div([
    #         'There was an error processing this file.'
    #     ])
    
    # create pandas dataframe that calculates string length and number of white spaces of string
    # only use first 20 characters of the string
    df = pd.DataFrame({'String': full_string[:20]}, index = [0])
    df['Name'] = filename
    df['Length'] = df['String'].apply(len)
    df['WhiteSpaces'] = df['String'].apply(lambda x: len(x) - len(x.replace(' ', '')))
    df['Language'] = language

    # sentiment analysis
    if sentiment == 'afinn':
        #df['Sentiment'] = df['String'].apply(lambda x: afinn.score(x))
        df['Sentiment'] = sentiment
    elif sentiment == 'vader':
        #df['Sentiment'] = df['String'].apply(lambda x: vader.polarity_scores(x))
        df['Sentiment'] = sentiment
    elif sentiment == 'syuzhet':
        #df['Sentiment'] = df['String'].apply(lambda x: syuzhet.get_sentiment(x))
        df['Sentiment'] = sentiment
    elif sentiment == 'avg_syuzhet_vader':
        #df['Sentiment'] = df['String'].apply(lambda x: (syuzhet.get_sentiment(x) + vader.polarity_scores(x)['compound'])/2)
        df['Sentiment'] = sentiment

    dict_0 = compute_metrics(full_string, language, sentiment)
    df = pd.DataFrame(dict_0)

    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        html.Hr(),  # horizontal line

        html.P(children=(full_string)),

        html.Hr(),  # horizontal line

        html.P(children=(text)),

        html.Hr(),  # horizontal line

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        dcc.Graph(
            figure={
                "data": [
                    {"y": df["Length"], "type": "bar", "name": filename},
                    {"y": mean_df["TITLE_LENGTH"], "type": "bar", "name": filename},
                ],
                "layout": {"title": "Average Price of Avocados"},
            },
        ),
    ])

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              Input('lang-dropdown', 'value'),
              Input('sent-dropdown', 'value'),
              Input('textarea-example', 'value'))
def update_output(list_of_contents, list_of_names, list_of_dates, language, sentiment, text):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d, language, sentiment, text) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        return children

if __name__ == '__main__':
    app.run(debug=True)