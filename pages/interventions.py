from dash import html, dcc, dash_table, Input, Output, State
import dash
import pandas as pd
from data import table_df

def layout():
    drivers_series = table_df["Drivers"].fillna("").astype(str)
    driver_set = set()
    for s in drivers_series:
        for token in [t.strip() for t in s.split(",") if t.strip()]:
            driver_set.add(token)
    driver_options = [{"label": d, "value": d} for d in sorted(driver_set)]

    preview_df = table_df.copy()
    preview_df["DriversPreview"] = preview_df["Drivers"].fillna("").apply(
        lambda s: "  ".join([f"`{t.strip()}`" for t in str(s).split(",") if t.strip()])
    )

    return html.Div([
        html.Div(className="interv-controls", children=[
            html.Div([
                dcc.Input(
                    id="interv-statecounty",
                    placeholder="Search state or county...",
                    type="text",
                    debounce=True,
                    style={"width":"320px","padding":"10px","borderRadius":"8px","border":"1px solid #e6e6e6"}
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id="interv-driver-filter",
                    options=driver_options,
                    placeholder="Filter by driver(s) (optional)",
                    multi=True,
                    style={"minWidth":"260px","maxWidth":"380px"}
                )
            ]),
            html.Div([

            ], style={"marginLeft":"auto"})
        ], style={"display":"flex","gap":"12px","alignItems":"center","marginTop":"6px","marginBottom":"12px"}),

        html.Div([
            dash_table.DataTable(
                id="tbl-interventions",
                columns=[
                    {"name":"State","id":"STATE"},
                    {"name":"County","id":"COUNTY"},
                    {"name":"Drivers","id":"DriversPreview", "presentation":"markdown"},
                    {"name":"High-Risk Share","id":"High-Risk Share"},
                    {"name":"Suggested Intervention","id":"SuggestedIntervention"}
                ],
                data=preview_df.to_dict("records"),
                page_size=10,
                sort_action="native",
                row_selectable="single",
                style_table={"overflowX":"auto","maxHeight":"62vh","overflowY":"auto","border":"none"},
                style_cell={
                    "textAlign":"left","padding":"10px","fontFamily":"Arial","fontSize":"13px",
                    "whiteSpace":"normal","height":"auto"
                },
                style_header={"backgroundColor":"transparent","fontWeight":"700","color":"#123e37","borderBottom":"1px solid rgba(0,0,0,0.06)"},
                style_data_conditional=[
                    {"if":{"row_index":"odd"}, "backgroundColor":"#fbfaf9"},
                    {"if":{"column_id":"High-Risk Share"}, "textAlign":"center", "fontWeight":"700"},
                ],
                tooltip_duration=None,
                style_as_list_view=True
            )
        ], style={"borderRadius":"10px","boxShadow":"0 8px 24px rgba(20,30,30,0.04)","padding":"12px","background":"#fff"}),

        dcc.Download(id="download-table"),
        html.Div(id="modal-container")
    ], style={"padding":"6px 0 24px 0"})

@dash.callback(
    Output("tbl-interventions","data"),
    Input("interv-statecounty","value"),
    Input("interv-driver-filter","value")
)
def filter_table(statecounty, selected_drivers):
    df2 = table_df.copy()
    if selected_drivers:
        sel = set(selected_drivers)
        def has_any_driver(s):
            s = str(s) if s is not None else ""
            tokens = [t.strip() for t in s.split(",") if t.strip()]
            return any(t in sel for t in tokens)
        df2 = df2[df2["Drivers"].apply(has_any_driver)]

    if statecounty and str(statecounty).strip():
        q = str(statecounty).strip().lower()
        mask = df2.apply(lambda r:
                         q in str(r.get("STATE","")).lower() or
                         q in str(r.get("COUNTY","")).lower(), axis=1)
        df2 = df2[mask]

    df2 = df2.copy()
    df2["DriversPreview"] = df2["Drivers"].fillna("").apply(
        lambda s: "  ".join([f"`{t.strip()}`" for t in str(s).split(",") if t.strip()])
    )
    df2["High-Risk Share"] = df2["High-Risk Share"].astype(str)

    return df2.to_dict("records")
@dash.callback(
    Output("download-table", "data"),
    Input("btn-download", "n_clicks"),
    State("tbl-interventions", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, table_data):
    if not n_clicks:
        return dash.no_update  # do nothing if button not clicked
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "interventions_high_risk_counties.csv", index=False)


@dash.callback(Output("modal-container","children"), Input("tbl-interventions","selected_rows"), State("tbl-interventions","data"))
def show_modal(selected_rows, rows):
    if not selected_rows:
        return ""
    r = rows[selected_rows[0]]
    return html.Div([
        html.Div(className="modal-backdrop", children=[
            html.Div(className="modal", children=[
                html.Div([
                    html.H3(f"{r.get('COUNTY','')} — {r.get('STATE','')}", style={"margin":"0 0 6px 0"}),
                    html.Div([html.Span("Drivers: ", style={"fontWeight":"700"}), html.Span(r.get("Drivers",""))]),
                    html.H4("Suggested Intervention", style={"marginTop":"12px"}),
                    html.P(r.get("SuggestedIntervention",""), style={"marginTop":"6px","lineHeight":"1.4"}),
                ], style={"padding":"6px 0"}),
                html.Div(style={"display":"flex","justifyContent":"flex-end","gap":"8px","marginTop":"10px"}, children=[
                    html.Button("Close", id="close-modal", n_clicks=0, className="btn-secondary"),
                ])
            ])
        ])
    ])

@dash.callback(Output("tbl-interventions","selected_rows"), Input("close-modal","n_clicks"), prevent_initial_call=True)
def close_modal(n):
    return []
