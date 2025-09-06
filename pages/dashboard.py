from dash import html, dcc
from data import df, make_map, fig_bar_state

initial_map = make_map(df)

def layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Jump to state:", style={"fontWeight": "700", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="dd-state",
                    options=[{"label": s, "value": s} for s in sorted(df["STATE"].dropna().unique())],
                    placeholder="Select a state",
                    style={"width": "320px"}
                )
            ], style={"display": "inline-block", "verticalAlign": "middle", "marginRight": "12px"}),

            html.Button(
                "Reset View",
                id="btn-reset",
                n_clicks=0,
                style={
                    "padding": "8px 12px",
                    "borderRadius": "6px",
                    "background": "#123e37",
                    "color": "#fff",
                    "border": "none",
                    "cursor": "pointer"
                }
            )
        ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "12px"}),

        html.Div([
            html.Div(
                dcc.Graph(id="risk-map", figure=initial_map, config={"scrollZoom": True}),
                className="map-visual"
            ),
        ], className="map-container"),

        html.Div([
            html.Div(
                dcc.Graph(id="bar-state-small", figure=fig_bar_state),
                className="statebar-card"
            )
        ], style={"marginTop": "14px", "display": "flex", "justifyContent": "center"})
    ], style={"padding": "6px 0 24px 0"})
