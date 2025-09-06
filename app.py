from dash import Dash, html, dcc, Input, Output, State, callback_context
import pages.dashboard as dashboard
import pages.charts as charts
import pages.interventions as interventions
from data import df, make_map
import traceback
import sys

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    "/assets/style.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="sidebar-collapsed", data=False),

    html.Div([
    html.Button("☰", id="btn-toggle-sidebar", className="sidebar-toggle", n_clicks=0, title="Toggle menu"),

    html.Div("NGO Dashboard", className="sidebar-title"),
    
    html.A([html.I(className="fa-solid fa-map-location-dot icon"), html.Span("Map", className="label")],
           href="/", id="link-dashboard", className="sidebar-link", **{"data-tooltip": "Map"}),

    html.A([html.I(className="fa-solid fa-chart-column icon"), html.Span("Charts & Analysis", className="label")],
           href="/charts", id="link-charts", className="sidebar-link", **{"data-tooltip": "Charts & Analysis"}),

    html.A([html.I(className="fa-solid fa-hand-holding-medical icon"), html.Span("Suggested Interventions", className="label")],
           href="/interventions", id="link-interv", className="sidebar-link", **{"data-tooltip": "Suggested Interventions"}),

    html.Div(className="sidebar-spacer"),
], id="sidebar", className="sidebar"),

    html.Div(id="page-content", className="content")
])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    try:
        if pathname is None or pathname == "/":
            return dashboard.layout()
        if pathname == "/charts":
            return charts.layout()
        if pathname == "/interventions":
            return interventions.layout()
        return html.Div([html.H3("404 - Page not found")], style={"padding":"20px"})
    except Exception:
        tb = traceback.format_exc()
        print("Error rendering page:", file=sys.stderr)
        print(tb, file=sys.stderr)
        return html.Div([
            html.H3("Error rendering page", style={"color":"#b00020"}),
            html.Pre(tb, style={"whiteSpace":"pre-wrap", "overflowX":"auto", "background":"#fff7f7",
                               "border":"1px solid #f1c2c2", "padding":"10px", "borderRadius":"6px"})
        ], style={"padding":"20px", "background":"#fff7f7", "border":"1px solid #f6d6d6", "borderRadius":"8px"})
@app.callback(
    Output("sidebar", "className"),
    Output("page-content", "className"),
    Input("btn-toggle-sidebar", "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=False
)
def toggle_sidebar(n_clicks, current_class):
    collapsed = False
    if current_class and "collapsed" in current_class:
        collapsed = True
    if n_clicks:
        collapsed = not collapsed
    if collapsed:
        return "sidebar collapsed", "content collapsed"
    return "sidebar", "content"
@app.callback(
    Output("link-dashboard", "className"),
    Output("link-charts", "className"),
    Output("link-interv", "className"),
    Input("url", "pathname"),
)
def highlight_active(pathname):
    base = "sidebar-link"
    a = base
    b = base
    c = base
    if pathname is None or pathname == "/":
        a = base + " active"
    elif pathname == "/charts":
        b = base + " active"
    elif pathname == "/interventions":
        c = base + " active"
    return a, b, c

@app.callback(
    Output("risk-map", "figure"),
    Output("dd-state", "value"),
    Input("bar-state-small", "clickData"),
    Input("dd-state", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=False
)
def zoom_to_state(bar_click, dd_value, n_reset):
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("btn-reset"):
        return make_map(df), None

    state = None
    if dd_value:
        state = dd_value
    elif bar_click and "points" in bar_click and bar_click["points"]:
        state = bar_click["points"][0].get("x")

    if not state:
        return make_map(df), None

    dff = df[df["STATE"] == state]
    if dff.empty:
        return make_map(df), None

    center = {"lat": dff["latitude"].mean(), "lon": dff["longitude"].mean()}
    fig = make_map(dff, zoom=5.5, center=center)
    return fig, state

if __name__ == "__main__":
    app.run(debug=True)
