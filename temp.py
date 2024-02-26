# app.py

import pandas as pd
from dash import Dash, dcc, html
import textdescriptives as td

file = open('data/temp.txt','r')

df = td.extract_metrics(text=file, spacy_model="en_core_web_lg", metrics=["readability", "coherence"])

df.to_csv('data/temp.csv')

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="FabulaNet Analytics"),
        html.P(
            children=(
                "Analyze your text data with descriptive statistics and visualizations. List of metrics available: "
            ),
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df["readability"],
                        "y": df["coherence"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Average Price of Avocados"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)