import dash
from dash import html
import dash_bootstrap_components as dbc

# Initialize the app - FIXED LINE
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.H1("IoT Dashboard Test", style={'textAlign': 'center'}),
    html.Hr(),
    html.P("If you can see this, Dash is working!", style={'textAlign': 'center'}),
])

# Run the app - NEW SYNTAX
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)