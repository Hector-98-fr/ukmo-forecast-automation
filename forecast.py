import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from pystac_client import Client
import planetary_computer
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime
import os


# =========================================================
# CONFIG
# =========================================================

SHAPEFILE_PATH = "data/Gaza_adm2.shp"


# =========================================================
# LOAD UKMO FORECAST DATA
# =========================================================

import s3fs
import xarray as xr
from datetime import datetime, timedelta, timezone

BASE = "met-office-atmospheric-model-data/global-deterministic-10km"

def find_latest_reference_time(fs, base, max_lookback_hours=240):
    now = datetime.now(timezone.utc)
    hour = (now.hour // 6) * 6
    candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    for _ in range(max_lookback_hours // 6):
        folder = candidate.strftime("%Y%m%dT%H00Z")
        path = f"{base}/{folder}"
        if fs.exists(path):
            return path, candidate
        candidate -= timedelta(hours=6)

    raise RuntimeError("No forecast folder found in lookback window")


def load_ukmo_data():

    print("Connecting to Met Office AWS Open Data bucket...")

    fs = s3fs.S3FileSystem(anon=True)

    latest_path, latest_ref_time = find_latest_reference_time(fs, BASE)

    print("Latest reference time:", latest_ref_time)

    assets_needed = [
        "temperature_at_screen_level",
        "wind_speed_at_10m",
        "wind_gust_at_10m",
        "precipitation_rate"
    ]

    datasets = []

    print("Downloading forecast variables...")

    for asset_name in assets_needed:

        files = sorted(fs.glob(f"{latest_path}/*-{asset_name}.nc"))

        print(f"  {asset_name}: {len(files)} horizon files")

        for f in files:
            with fs.open(f) as fobj:
                ds = xr.open_dataset(fobj)
                datasets.append(ds)

    print("Download complete.")

    return datasets

# =========================================================
# PROCESS FORECAST
# =========================================================

def process_forecast(datasets):

    print(len(datasets))
    
    print("Processing forecast variables...")

    temp_ds = [
        ds[['air_temperature']].sel(
            latitude=slice(30, 33),
            longitude=slice(33, 36)
        )
        for ds in datasets
        if 'air_temperature' in ds.data_vars
    ]

    ws_ds = [
        ds[['wind_speed']].sel(
            latitude=slice(30, 33),
            longitude=slice(33, 36)
        )
        for ds in datasets
        if 'wind_speed' in ds.data_vars
    ]

    wind_ds = [
        ds[['wind_speed_of_gust']].sel(
            latitude=slice(30, 33),
            longitude=slice(33, 36)
        )
        for ds in datasets
        if 'wind_speed_of_gust' in ds.data_vars
    ]

    precip_ds = [
        ds[['lwe_precipitation_rate']].sel(
            latitude=slice(30, 33),
            longitude=slice(33, 36)
        )
        for ds in datasets
        if 'lwe_precipitation_rate' in ds.data_vars
    ]

    temp_all = xr.concat(temp_ds, dim="time")
    ws_all = xr.concat(ws_ds, dim="time")
    wind_all = xr.concat(wind_ds, dim="time")
    precip_all = xr.concat(precip_ds, dim="time")

    ds_all = xr.merge(
        [temp_all, ws_all, wind_all, precip_all],
        compat='override'
    )

    # Unit conversions
    ds_all['air_temperature'] = (
        ds_all['air_temperature'] - 273.15
    )

    ds_all['wind_speed'] = (
        ds_all['wind_speed'] * 3.6
    )

    ds_all['wind_speed_of_gust'] = (
        ds_all['wind_speed_of_gust'] * 3.6
    )

    ds_all['lwe_precipitation_rate'] = (
        ds_all['lwe_precipitation_rate'] * 86400 * 1000
    )

    ds_all = ds_all.sortby("time")

    # Daily aggregation
    daily_temp = ds_all["air_temperature"].resample(
        time="1D"
    ).max()

    daily_ws = ds_all["wind_speed"].resample(
        time="1D"
    ).mean()

    daily_wind = ds_all["wind_speed_of_gust"].resample(
        time="1D"
    ).max()

    daily_precip = ds_all["lwe_precipitation_rate"].resample(
        time="1D"
    ).sum()

    daily_ds = xr.merge([
        daily_temp,
        daily_ws,
        daily_wind,
        daily_precip
    ])

    # Keep first 7 days
    daily_ds = daily_ds.isel(time=slice(0, 7))

    print("Forecast processing complete.")

    return daily_ds


# =========================================================
# GOVERNORATE WEIGHTED TABLES
# =========================================================

def calculate_governorate_tables(
    ds,
    shapefile_path,
    governorate_column='adm2_name'
):

    print("Loading shapefile...")

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')

    lats = ds.latitude.values
    lons = ds.longitude.values

    lat_edges = np.zeros(len(lats) + 1)
    lon_edges = np.zeros(len(lons) + 1)

    lat_edges[1:-1] = (
        lats[1:] + lats[:-1]
    ) / 2

    lon_edges[1:-1] = (
        lons[1:] + lons[:-1]
    ) / 2

    lat_edges[0] = (
        lats[0] - (lats[1] - lats[0]) / 2
    )

    lat_edges[-1] = (
        lats[-1] + (lats[-1] - lats[-2]) / 2
    )

    lon_edges[0] = (
        lons[0] - (lons[1] - lons[0]) / 2
    )

    lon_edges[-1] = (
        lons[-1] + (lons[-1] - lons[-2]) / 2
    )

    target_crs = 'EPSG:32636'

    gdf_proj = gdf.to_crs(target_crs)

    grid_cells = []

    for i in range(len(lats)):
        for j in range(len(lons)):

            grid_cells.append({
                'lat_idx': i,
                'lon_idx': j,
                'geometry': box(
                    lon_edges[j],
                    lat_edges[i],
                    lon_edges[j+1],
                    lat_edges[i+1]
                )
            })

    grid_gdf = gpd.GeoDataFrame(
        grid_cells,
        crs='EPSG:4326'
    ).to_crs(target_crs)

    gov_fractions = {}

    print("Computing area-weighted intersections...")

    for _, gov in gdf_proj.iterrows():

        gov_name = gov[governorate_column]
        gov_geom = gov.geometry

        intersections = grid_gdf.geometry.intersection(
            gov_geom
        )

        areas = intersections.area
        total_area = gov_geom.area

        cells = []

        for k in range(len(grid_gdf)):

            if areas.iloc[k] > 0:

                cells.append({
                    'lat_idx': grid_gdf.iloc[k]['lat_idx'],
                    'lon_idx': grid_gdf.iloc[k]['lon_idx'],
                    'fraction': areas.iloc[k] / total_area
                })

        gov_fractions[gov_name] = cells

    temp = ds["air_temperature"].values
    ws = ds["wind_speed"].values
    wind = ds["wind_speed_of_gust"].values
    precip = ds["lwe_precipitation_rate"].values
    times = ds.time.values

    gov_names = list(gov_fractions.keys())

    temp_out = np.zeros((len(times), len(gov_names)))
    ws_out = np.zeros((len(times), len(gov_names)))
    wind_out = np.zeros((len(times), len(gov_names)))
    precip_out = np.zeros((len(times), len(gov_names)))

    print("Computing weighted averages...")

    for g, gov_name in enumerate(gov_names):

        cells = gov_fractions[gov_name]

        for t in range(len(times)):

            t_val = 0
            ws_val = 0
            w_val = 0
            p_val = 0

            for cell in cells:

                i = cell['lat_idx']
                j = cell['lon_idx']
                f = cell['fraction']

                t_val += f * temp[t, i, j]
                ws_val += f * ws[t, i, j]
                w_val += f * wind[t, i, j]
                p_val += f * precip[t, i, j]

            temp_out[t, g] = t_val
            ws_out[t, g] = ws_val
            wind_out[t, g] = w_val
            precip_out[t, g] = p_val

    df_temp = pd.DataFrame(
        temp_out,
        index=times,
        columns=gov_names
    )

    df_ws = pd.DataFrame(
        ws_out,
        index=times,
        columns=gov_names
    )

    df_wind = pd.DataFrame(
        wind_out,
        index=times,
        columns=gov_names
    )

    df_precip = pd.DataFrame(
        precip_out,
        index=times,
        columns=gov_names
    )

    print("Governorate tables complete.")

    return (
        df_temp,
        df_ws,
        df_wind,
        df_precip
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("Starting UKMO forecast pipeline...")

    # Create dated output folder
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    output_dir = os.path.join(
        "outputs",
        run_date
    )

    os.makedirs(output_dir, exist_ok=True)

    print("Output directory:", output_dir)

    # Load and process forecast
    datasets = load_ukmo_data()

    daily_ds = process_forecast(datasets)

    # Governorate aggregation
    (
        df_temp,
        df_ws,
        df_wind,
        df_precip
    ) = calculate_governorate_tables(
        daily_ds,
        SHAPEFILE_PATH
    )

        # =========================================================
    # WEEKLY TOTAL PRECIPITATION (PER GOVERNORATE)
    # =========================================================

    df_precip_total = (
        df_precip
        .sum(axis=0)
        .to_frame(name="Total_7day_Precip")
        .reset_index()
    )

    df_precip_total = df_precip_total.rename(
        columns={"index": "District"}
    )

    # =========================================================
    # ROUNDING
    # =========================================================

    df_temp = df_temp.round(1)

    df_ws = df_ws.round(0).astype(int)

    df_wind = df_wind.round(0).astype(int)

    df_precip = df_precip.round(1).astype(int)

    df_precip_total["Total_7day_Precip"] = (
        df_precip_total["Total_7day_Precip"]
        .round(1)
        .astype(int)
    )

    # =========================================================
    # CONVERT TO LONG FORMAT
    # =========================================================

    def convert_to_long_format(df, value_name):

        df_long = df.reset_index()

        # First column is the date
        df_long.columns = ["Date"] + list(df_long.columns[1:])

        df_long = df_long.melt(
            id_vars="Date",
            var_name="District",
            value_name=value_name
        )

        # Format dates as 2-Jun, 3-Jun, etc.
        df_long["Date"] = pd.to_datetime(
            df_long["Date"]
        ).dt.strftime("%-d-%b")

        return df_long

    df_temp_long = convert_to_long_format(
        df_temp,
        "Temperature"
    )

    df_ws_long = convert_to_long_format(
        df_ws,
        "WindSpeed"
    )

    df_wind_long = convert_to_long_format(
        df_wind,
        "Wind"
    )

    df_precip_long = convert_to_long_format(
        df_precip,
        "Precipitation"
    )

    # =====================================================
    # EXPORT CSV FILES
    # =====================================================

    print("Exporting CSV files...")

    df_temp_long.to_csv(
        os.path.join(output_dir, "df_temp.csv"),
        index=False
    )

    df_ws_long.to_csv(
        os.path.join(output_dir, "df_ws.csv"),
        index=False
    )

    df_wind_long.to_csv(
        os.path.join(output_dir, "df_wind.csv"),
        index=False
    )

    df_precip_long.to_csv(
        os.path.join(output_dir, "df_precip.csv"),
        index=False
    )

    df_precip_total.to_csv(
        os.path.join(
            output_dir,
            "df_precip_total.csv"
        ),
        index=False
    )

    print("All CSV files exported successfully.")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
