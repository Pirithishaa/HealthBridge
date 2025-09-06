from dash import html, dcc, Input, Output
import dash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data import df, color_map, high_risk_pct, avg_risk_score, food_desert_count, poor_vehicle_access

def empty_figure(title="No data"):
    fig = go.Figure()
    fig.update_layout(title=title, height=340,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=12, r=12, t=36, b=12))
    fig.add_annotation(text=title, xref="paper", yref="paper", showarrow=False,
                       x=0.5, y=0.5, font=dict(size=14, color="#666"))
    return fig

def layout():
    rs_min = float(df["RiskScore"].min(skipna=True)) if "RiskScore" in df.columns else 0.0
    rs_max = float(df["RiskScore"].max(skipna=True)) if "RiskScore" in df.columns else 1.0
    states = sorted(df["STATE"].dropna().unique()) if "STATE" in df.columns else []

    kpi_cards = html.Div([
        html.Div([html.H6("High Risk %"), html.H3(f"{high_risk_pct}%")], className="card"),
        html.Div([html.H6("Average Risk Score"), html.H3(f"{avg_risk_score}")], className="card"),
        html.Div([html.H6("Food Desert Count"), html.H3(f"{int(food_desert_count):,}")], className="card"),
        html.Div([html.H6("Poor Vehicle Access"), html.H3(f"{int(poor_vehicle_access):,}")], className="card"),
    ], className="kpi-row")

    controls = html.Div([
        html.Div([
            html.Label("State"),
            dcc.Dropdown(
                id="charts-state",
                options=[{"label": s, "value": s} for s in states],
                placeholder="All states",
                value=None,
                clearable=True,
                style={"width":"260px"}
            )
        ], style={"minWidth":"220px"}),

        html.Div([
            html.Label("Risk Categories"),
            dcc.Checklist(
                id="charts-riskcat",
                options=[{"label": k, "value": k} for k in color_map.keys()],
                value=list(color_map.keys()),
                inline=True
            )
        ], style={"minWidth":"320px"}),

        html.Div([
            html.Div(
                dcc.RangeSlider(
                    id="charts-rangescore",
                    min=rs_min,
                    max=rs_max,
                    step=0.1 if rs_max - rs_min > 0 else 1,
                    value=[rs_min, rs_max],
                    marks={int(rs_min): str(int(rs_min)), int(rs_max): str(int(rs_max))},
                    tooltip={"placement":"bottom", "always_visible": False},
                    persistence=True,
                    persistence_type="session",
                ),
                style={"display":"none"}
            ),
            html.Button("Reset Filters", id="charts-reset", n_clicks=0,
                        style={"marginLeft":"8px","padding":"8px 12px","borderRadius":"6px","background":"#123e37","color":"#fff","border":"none","cursor":"pointer"})
        ], style={"display":"flex","alignItems":"center","gap":"8px","flex":"1 1 320px","justifyContent":"flex-end"})
    ], className="controls")

    charts_grid = html.Div([
        html.Div(dcc.Graph(id="chart-poverty", figure=empty_figure("Loading...")), className="chart-card"),
        html.Div(dcc.Graph(id="chart-food", figure=empty_figure("Loading...")), className="chart-card"),
        html.Div(dcc.Graph(id="chart-unemp", figure=empty_figure("Loading...")), className="chart-card"),
        html.Div(dcc.Graph(id="chart-statebar", figure=empty_figure("Loading...")), className="chart-card"),
    ], className="charts-grid-grid")

    return html.Div([
        kpi_cards,
        controls,
        charts_grid
    ], style={"padding":"6px 0 24px 0"})

@dash.callback(
    Output("chart-poverty","figure"),
    Output("chart-food","figure"),
    Output("chart-unemp","figure"),
    Output("chart-statebar","figure"),
    Input("charts-state","value"),
    Input("charts-riskcat","value"),
    Input("charts-rangescore","value")
)
def update_charts(state, riskcats, rangescore):
    dff = df.copy()

    if state:
        if "STATE" in dff.columns:
            dff = dff[dff["STATE"] == state]
        else:
            dff = dff.iloc[0:0]

    if riskcats:
        if "RiskCategory" in dff.columns:
            dff = dff[dff["RiskCategory"].isin(riskcats)]
        else:
            dff = dff.iloc[0:0]

    if rangescore and isinstance(rangescore, (list, tuple)) and len(rangescore) == 2:
        lo, hi = rangescore[0], rangescore[1]
        if "RiskScore" in dff.columns:
            try:
                dff = dff[dff["RiskScore"].between(lo, hi, inclusive="both")]
            except TypeError:
                dff = dff[(dff["RiskScore"] >= lo) & (dff["RiskScore"] <= hi)]
        else:
            dff = dff.iloc[0:0]

    try:
        if dff.empty:
            fig1 = empty_figure("No data for selected filters")
        else:
            pov_df = dff.groupby("RiskCategory")["poverty_rate"].mean().reset_index().fillna(0)
            fig1 = px.bar(pov_df, x="RiskCategory", y="poverty_rate", color="RiskCategory",
                          color_discrete_map=color_map, title="Avg Poverty Rate by Risk Category")
            fig1.update_layout(height=360)
    except Exception:
        fig1 = empty_figure("Error building poverty chart")

    try:
        if dff.empty:
            fig2 = empty_figure("No data for selected filters")
        else:
            dff2 = dff.copy()
            dff2["FoodDesert"] = dff2["low_access_pop"].apply(lambda x: "Yes" if str(x) == "1" else "No")
            food_df = dff2.groupby(["FoodDesert","RiskCategory"]).size().reset_index(name="count")
            fig2 = px.bar(food_df, x="FoodDesert", y="count", color="RiskCategory", barmode="group",
                          color_discrete_map=color_map, title="Risk Distribution by Food Desert")
            fig2.update_layout(height=360)
    except Exception:
        fig2 = empty_figure("Error building food desert chart")

    try:
        if dff.empty:
            fig3 = empty_figure("No data for selected filters")
        else:
            fig3 = px.scatter(dff, x="unemployment", y="RiskScore", color="RiskCategory",
                              title="Unemployment vs Risk Score", color_discrete_map=color_map)
            fig3.update_layout(height=360)
    except Exception:
        fig3 = empty_figure("Error building unemployment chart")

    try:
        if dff.empty:
            fig4 = empty_figure("No data for selected filters")
        else:
            risk_by_state = (dff.groupby("STATE")["RiskCategory"]
                             .apply(lambda s: (s == "High Risk").mean() * 100)
                             .reset_index(name="HighRisk%").sort_values("HighRisk%"))
            if risk_by_state.empty:
                fig4 = empty_figure("No data for selected filters")
            else:
                fig4 = px.bar(risk_by_state, x="STATE", y="HighRisk%", title="High Risk % by State",
                              color_discrete_sequence=["#e63946"])
                fig4.update_layout(height=360)
    except Exception:
        fig4 = empty_figure("Error building state bar chart")

    return fig1, fig2, fig3, fig4

@dash.callback(
    Output("charts-state","value"),
    Output("charts-riskcat","value"),
    Output("charts-rangescore","value"),
    Input("charts-reset","n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n):
    rs_min = float(df["RiskScore"].min(skipna=True)) if "RiskScore" in df.columns else 0.0
    rs_max = float(df["RiskScore"].max(skipna=True)) if "RiskScore" in df.columns else 1.0
    default_riskcats = list(color_map.keys())
    return None, default_riskcats, [rs_min, rs_max]
