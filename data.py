import pandas as pd
import plotly.express as px

df = pd.read_csv("C:/Users/perumal/Desktop/cts project/ngo/ngo_8states_clean.csv")
df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)

for col in ["poverty_rate","unemployment","uninsured","low_access_pop","vehicle_access","RiskScore","latitude","longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

color_map = {"High Risk": "#e63946", "Medium Risk": "#f4a261", "Low Risk": "#457b9d"}

high_risk_pct = round((df["RiskCategory"].eq("High Risk").mean()) * 100, 2)
avg_risk_score = round(df["RiskScore"].mean(), 2)
food_desert_count = pd.to_numeric(df["low_access_pop"], errors="coerce").fillna(0).sum()
poor_vehicle_access = pd.to_numeric(df["vehicle_access"], errors="coerce").fillna(0).sum()

def make_map(dff, zoom=3, center={"lat":37.8,"lon":-96}):
    fig = px.scatter_mapbox(
        dff, lat="latitude", lon="longitude",
        color="RiskCategory", size="RiskScore",
        color_discrete_map=color_map,
        hover_data={"STATE": True,"COUNTY": True,"RiskScore": ':.2f',
                    "poverty_rate": ':.2f',"unemployment": ':.2f',"uninsured": ':.2f',
                    "latitude": False,"longitude": False},
        zoom=zoom, height=520, center=center
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":30,"l":0,"b":0},
        title="Geographic Distribution of Risk",
        uirevision="keep"
    )
    return fig

risk_by_state = (
    df.groupby("STATE")["RiskCategory"]
      .apply(lambda s: (s == "High Risk").mean() * 100)
      .reset_index(name="HighRisk%")
      .sort_values("HighRisk%")
)
fig_bar_state = px.bar(
    risk_by_state, x="STATE", y="HighRisk%",
    title="High Risk % by State", color_discrete_sequence=["#e63946"]
)

df["FoodDesert"] = df["low_access_pop"].apply(lambda x: "Yes" if str(x) == "1" else "No")
food_risk = df.groupby(["FoodDesert", "RiskCategory"]).size().reset_index(name="count")
fig_food = px.bar(
    food_risk, x="FoodDesert", y="count", color="RiskCategory",
    title="Risk Distribution by Food Desert Status",
    barmode="group", color_discrete_map=color_map
)

fig_unemp = px.scatter(
    df, x="unemployment", y="RiskScore", color="RiskCategory",
    title="Unemployment vs Risk Score", color_discrete_map=color_map
)

fig_poverty = px.bar(
    df.groupby("RiskCategory")["poverty_rate"].mean().reset_index(),
    x="RiskCategory", y="poverty_rate", color="RiskCategory",
    title="Poverty Rate Across Risk Categories", color_discrete_map=color_map
)

for fig in [fig_bar_state, fig_food, fig_unemp, fig_poverty]:
    fig.update_layout(height=300, margin=dict(t=40,b=30,l=40,r=20),
                      title_font=dict(size=14), font=dict(size=10),
                      legend=dict(font=dict(size=9)))
    fig.update_xaxes(title_font=dict(size=11), tickfont=dict(size=9))
    fig.update_yaxes(title_font=dict(size=11), tickfont=dict(size=9))

county = (
    df.groupby(["STATE","COUNTY"])
      .agg(
          tracts=("tract_id","nunique"),
          high_risk_share=("RiskCategory", lambda s: (s == "High Risk").mean()),
          pov=("poverty_rate","mean"),
          unemp=("unemployment","mean"),
          unins=("uninsured","mean"),
          fd_share=("low_access_pop", lambda s: (s.fillna(0) == 1).mean()),
          veh_share=("vehicle_access",  lambda s: (s.fillna(0) == 1).mean()),
      ).reset_index()
)

THR = {"pov":0.25, "unemp":0.10, "unins":0.15, "fd_share":0.30, "veh_share":0.20}
PHRASES = {
    "pov":"food bank + SNAP/WIC outreach",
    "unemp":"8–12 wk job training + placement",
    "unins":"Medicaid signup + pop-up clinic",
    "fd_share":"mobile fresh-food markets",
    "veh_share":"grocery/clinic shuttle service"
}
TAGS = {"pov":"Poverty","unemp":"Unemployment","unins":"Uninsured",
        "fd_share":"Food Desert","veh_share":"Poor Vehicle Access"}

def nz(x): return 0.0 if pd.isna(x) else float(x)

def top_driver_suggestion(row):
    scores = {k: nz(row[k]) / THR[k] for k in THR}
    over = [k for k,v in scores.items() if v >= 1.0]
    if over:
        drivers = sorted(over, key=lambda k: scores[k], reverse=True)[:2]
    else:
        drivers = sorted(scores, key=scores.get, reverse=True)[:2]
    suggestion = "Mobile hub: " + " + ".join(PHRASES[d] for d in drivers)
    drivers_str = ", ".join(TAGS[d] for d in drivers)
    return suggestion, drivers_str

county[["SuggestedIntervention","Drivers"]] = county.apply(
    lambda r: pd.Series(top_driver_suggestion(r)), axis=1
)

hi_county = county.loc[county["high_risk_share"] >= 0.5].copy()
table_df = (
    hi_county
    .assign(**{"High-Risk Share": (hi_county["high_risk_share"]*100).round(1).astype(str) + "%"} )
)[["STATE","COUNTY","Drivers","High-Risk Share","SuggestedIntervention"]].sort_values(["STATE","COUNTY"])
