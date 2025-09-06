# ngo_dashboard.py  (fixed)
from dash import Dash, html, dcc, Input, Output, State, callback_context
import traceback
import sys
import importlib

# data helpers used by callbacks
from data import df, make_map

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    "/assets/style.css",
]

def register_dash(flask_app):
    """
    Create and mount Dash app on the provided Flask app.
    Mount path: /ngo/dashboard/
    """
    # 1) Create the Dash app FIRST so module-level @callback in pages bind correctly
    ngo_dash = Dash(
        __name__,
        server=flask_app,
        url_base_pathname="/ngo/dashboard/",
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
    )

    # 2) Import page modules AFTER app exists (critical for callbacks to register)
    try:
        dashboard = importlib.import_module("pages.dashboard")
        charts = importlib.import_module("pages.charts")
        interventions = importlib.import_module("pages.interventions")
    except Exception:
        # Helpful trace in the UI if a page import fails
        tb = traceback.format_exc()
        print("Error importing page modules:\n", tb, file=sys.stderr)

        def _err_layout():
            return html.Div(
                [
                    html.H3("Error importing page modules", style={"color": "#b00020"}),
                    html.Pre(
                        tb,
                        style={
                            "whiteSpace": "pre-wrap",
                            "overflowX": "auto",
                            "background": "#fff7f7",
                            "border": "1px solid #f1c2c2",
                            "padding": "10px",
                            "borderRadius": "6px",
                        },
                    ),
                ],
                style={
                    "padding": "20px",
                    "background": "#fff7f7",
                    "border": "1px solid #f6d6d6",
                    "borderRadius": "8px",
                },
            )

        class _Fallback:
            layout = staticmethod(_err_layout)

        dashboard = charts = interventions = _Fallback()

    # 3) App shell (sidebar + placeholder for pages)
    ngo_dash.layout = html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="sidebar-collapsed", data=False),

            # Sidebar
            html.Div(
                [
                    html.Button(
                        "☰",
                        id="btn-toggle-sidebar",
                        className="sidebar-toggle",
                        n_clicks=0,
                        title="Toggle menu",
                    ),
                    html.Div("NGO Dashboard", className="sidebar-title"),

                    html.A(
                        [html.I(className="fa-solid fa-map-location-dot icon"), html.Span("Map", className="label")],
                        href="/ngo/dashboard/",
                        id="link-dashboard",
                        className="sidebar-link",
                        **{"data-tooltip": "Map"},
                    ),
                    html.A(
                        [html.I(className="fa-solid fa-chart-column icon"), html.Span("Charts & Analysis", className="label")],
                        href="/ngo/dashboard/charts",
                        id="link-charts",
                        className="sidebar-link",
                        **{"data-tooltip": "Charts & Analysis"},
                    ),
                    html.A(
                        [html.I(className="fa-solid fa-hand-holding-medical icon"), html.Span("Suggested Interventions", className="label")],
                        href="/ngo/dashboard/interventions",
                        id="link-interv",
                        className="sidebar-link",
                        **{"data-tooltip": "Suggested Interventions"},
                    ),

                    # CSV Download (visible on all pages)
                    html.Div(
                        [
                            html.Button(
                                "Download CSV",
                                id="btn-download",
                                n_clicks=0,
                                className="sidebar-link button-link",
                                title="Download filtered dataset (full if no filters)",
                            ),
                            dcc.Download(id="download-data"),
                        ],
                        style={"padding": "10px 14px"},
                    ),

                    html.Div(className="sidebar-spacer"),
                ],
                id="sidebar",
                className="sidebar",
            ),

            # Page container
            html.Div(id="page-content", className="content"),
        ]
    )

    # ----------- Routing / Page rendering -----------
    @ngo_dash.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page(pathname):
        try:
            if not pathname:
                return dashboard.layout()

            # Trim base
            if pathname.startswith("/ngo/dashboard"):
                sub = pathname[len("/ngo/dashboard"):] or "/"
            else:
                sub = pathname

            if sub in ("", "/"):
                return dashboard.layout()
            if sub == "/charts":
                return charts.layout()
            if sub == "/interventions":
                return interventions.layout()
            return html.Div([html.H3("404 - Page not found")], style={"padding": "20px"})
        except Exception:
            tb = traceback.format_exc()
            print("Error rendering page:", file=sys.stderr)
            print(tb, file=sys.stderr)
            return html.Div(
                [
                    html.H3("Error rendering page", style={"color": "#b00020"}),
                    html.Pre(
                        tb,
                        style={
                            "whiteSpace": "pre-wrap",
                            "overflowX": "auto",
                            "background": "#fff7f7",
                            "border": "1px solid #f1c2c2",
                            "padding": "10px",
                            "borderRadius": "6px",
                        },
                    ),
                ],
                style={
                    "padding": "20px",
                    "background": "#fff7f7",
                    "border": "1px solid #f6d6d6",
                    "borderRadius": "8px",
                },
            )

    # ----------- Sidebar collapse -----------
    @ngo_dash.callback(
        Output("sidebar", "className"),
        Output("page-content", "className"),
        Input("btn-toggle-sidebar", "n_clicks"),
        State("sidebar", "className"),
        prevent_initial_call=False,
    )
    def toggle_sidebar(n_clicks, current_class):
        collapsed = bool(current_class and "collapsed" in current_class)
        if n_clicks:
            collapsed = not collapsed
        return ("sidebar collapsed", "content collapsed") if collapsed else ("sidebar", "content")

    # ----------- Highlight active menu -----------
    @ngo_dash.callback(
        Output("link-dashboard", "className"),
        Output("link-charts", "className"),
        Output("link-interv", "className"),
        Input("url", "pathname"),
    )
    def highlight_active(pathname):
        base = "sidebar-link"
        a = b = c = base
        sub = (pathname or "/")
        if pathname and pathname.startswith("/ngo/dashboard"):
            sub = pathname[len("/ngo/dashboard"):] or "/"
        if sub in ("", "/"):
            a = base + " active"
        elif sub == "/charts":
            b = base + " active"
        elif sub == "/interventions":
            c = base + " active"
        return a, b, c

    # ----------- Map zoom callback -----------
    @ngo_dash.callback(
        Output("risk-map", "figure"),
        Output("dd-state", "value"),
        Input("bar-state-small", "clickData"),
        Input("dd-state", "value"),
        Input("btn-reset", "n_clicks"),
        prevent_initial_call=False,
    )
    def zoom_to_state(bar_click, dd_value, n_reset):
        ctx = callback_context
        # Reset → full map
        if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("btn-reset"):
            return make_map(df), None

        # Derive selected state
        state = dd_value
        if not state and bar_click and "points" in bar_click and bar_click["points"]:
            state = bar_click["points"][0].get("x")

        if not state:
            return make_map(df), None

        dff = df[df["STATE"] == state]
        if dff.empty:
            return make_map(df), None

        center = {"lat": dff["latitude"].mean(), "lon": dff["longitude"].mean()}
        fig = make_map(dff, zoom=5.5, center=center)
        return fig, state

    # ----------- CSV Download -----------
    @ngo_dash.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks):
        # If you later add filters in dcc.Store, read them here and filter df
        return dcc.send_data_frame(df.to_csv, "ngo_8states_clean.csv", index=False)

    # ----------- Intervention details -----------
    # Expects components in interventions.layout:
    #   dcc.Dropdown(id="intervention-dropdown")
    #   html.Div(id="intervention-details")
    @ngo_dash.callback(
        Output("intervention-details", "children"),
        Input("intervention-dropdown", "value"),
        prevent_initial_call=False,
    )
    def show_intervention_details(selected):
        # Works whether the column is named "Intervention" or "intervention"
        if not selected:
            return "Select an intervention to view details."
        col = "Intervention" if "Intervention" in df.columns else ("intervention" if "intervention" in df.columns else None)
        if not col:
            return html.Div("Intervention column not found in data.")
        rows = df[df[col] == selected]
        if rows.empty:
            return html.Div("No details available for the selected intervention.")

        row = rows.iloc[0].to_dict()
        # Pick some optional keys if they exist
        def val(k, default="—"):
            return row.get(k, default)

        return html.Div(
            [
                html.H4(selected),
                html.P(f"Description: {val('description')}"),
                html.P(f"Effectiveness: {val('effectiveness')}"),
                html.P(f"Target Area: {val('target_area')}"),
                html.P(f"Estimated Cost: {val('cost')}"),
                html.P(f"Evidence Level: {val('evidence_level')}"),
            ],
            style={"padding": "10px 0"},
        )

    return ngo_dash
