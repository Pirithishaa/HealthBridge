import pandas as pd
import geopandas as gpd
import os

svi = pd.read_csv("SVI_2022_US.csv", dtype=str)

svi = svi.astype({
    "FIPS": str,
    "E_TOTPOP": float,
    "E_POV150": float,
    "E_UNEMP": float,
    "E_NOHSDP": float,
    "E_UNINSUR": float
})

svi = svi[["FIPS","STATE","COUNTY","E_TOTPOP","E_POV150","E_UNEMP","E_NOHSDP","E_UNINSUR"]]
svi.rename(columns={
    "FIPS":"tract_id",
    "E_POV150":"poverty_count",
    "E_UNEMP":"unemployed_count",
    "E_NOHSDP":"no_hs_count",
    "E_UNINSUR":"uninsured_count"
}, inplace=True)

svi["poverty_rate"] = svi["poverty_count"] / svi["E_TOTPOP"]
svi["unemployment"] = svi["unemployed_count"] / svi["E_TOTPOP"]
svi["education"] = svi["no_hs_count"] / svi["E_TOTPOP"]
svi["uninsured"] = svi["uninsured_count"] / svi["E_TOTPOP"]

food = pd.read_csv("Food Access Research Atlas.csv", dtype=str)
food = food[["CensusTract","State","County","LILATracts_halfAnd10","LILATracts_Vehicle"]]
food.rename(columns={
    "CensusTract":"tract_id",
    "LILATracts_halfAnd10":"low_access_pop",
    "LILATracts_Vehicle":"vehicle_access"
}, inplace=True)


selected_states = {
    "California":"06",
    "Texas":"48",
    "Florida":"12",
    "New York":"36",
    "Illinois":"17",
    "Pennsylvania":"42",
    "Mississippi":"28",
    "West Virginia":"54"
}

svi["tract_id"] = svi["tract_id"].str.zfill(11)
food["tract_id"] = food["tract_id"].str.zfill(11)

svi_sel = svi[svi["STATE"].isin(selected_states.keys())]
food_sel = food[food["State"].isin(selected_states.keys())]
merged = svi_sel.merge(food_sel, on="tract_id", how="inner")


merged["RiskScore"] = (
    0.4*merged["poverty_rate"].fillna(0) +
    0.2*merged["unemployment"].fillna(0) +
    0.2*merged["education"].fillna(0) +
    0.2*merged["uninsured"].fillna(0)
) * 100


q1 = merged["RiskScore"].quantile(0.33)
q2 = merged["RiskScore"].quantile(0.66)

def categorize(score):
    if score >= q2:
        return "High Risk"
    elif score >= q1:
        return "Medium Risk"
    else:
        return "Low Risk"

merged["RiskCategory"] = merged["RiskScore"].apply(categorize)


all_states_gdf = []

for state, fips in selected_states.items():
    shp_path = f"tl_2022_{fips}_tract.shp"  
    if not os.path.exists(shp_path):
        print(f"Shapefile missing: {shp_path}")
        continue
    
    tracts = gpd.read_file(shp_path)
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    
    state_merge = merged[merged["STATE"] == state]
    geo_merge = tracts.merge(state_merge, left_on="GEOID", right_on="tract_id", how="inner")
    all_states_gdf.append(geo_merge)


if all_states_gdf:
    final_gdf = pd.concat(all_states_gdf, ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry="geometry", crs="EPSG:4269")

    final_gdf = final_gdf.to_crs(epsg=4326)

    final_gdf["longitude"] = final_gdf.geometry.centroid.x
    final_gdf["latitude"] = final_gdf.geometry.centroid.y
    drop_cols = ["State","County","STATEFP","COUNTYFP","TRACTCE","NAME","NAMELSAD",
                 "MTFCC","FUNCSTAT","ALAND","AWATER","INTPTLAT","INTPTLON"]
    final_gdf = final_gdf.drop(columns=[c for c in drop_cols if c in final_gdf.columns], errors="ignore")
    
    final_gdf.drop(columns="geometry").to_csv("ngo_8states_clean.csv", index=False)

    final_gdf.to_file("ngo_8states.geojson", driver="GeoJSON")

    print("Exported")
else:
    print("No shapefiles processed")
