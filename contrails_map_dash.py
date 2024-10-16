import json
from google.cloud import storage
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, timezone
import re
import concurrent.futures

import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import dash_daq as daq

from shapely.geometry import LineString


def list_blobs_with_prefix(bucket_name, prefix):
    storage_client = storage.Client()
    return storage_client.list_blobs(bucket_name, prefix=prefix)

def extract_timestamp_from_path(path):
    # For ADSB data
    adsb_match = re.search(r'/(\d+)/adsb\.json$', path)
    if adsb_match:
        unix_timestamp = int(adsb_match.group(1))
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    
    # For contrail data
    contrail_match = re.search(r'/(\d{4})/(\d{3})/\d{2}/(\d+)/', path)
    if contrail_match:
        year, doy, unix_timestamp = contrail_match.groups()
        unix_timestamp = int(unix_timestamp)
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    
    return None

def load_adsb_data(bucket_name, date, min_altitude):
    adsb_data = []
    year = date.year
    doy = date.timetuple().tm_yday

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def process_blob(blob):
        if blob.name.endswith('adsb.json'):
            try:
                data = json.loads(blob.download_as_text())
                for entry in data:
                    if 'alt_geom' in entry and entry['alt_geom'] is not None and entry['alt_geom'] >= min_altitude:
                        entry_time_str = entry.get('time')
                        if entry_time_str:
                            entry_timestamp = datetime.fromisoformat(entry_time_str)
                            entry_timestamp = entry_timestamp.replace(tzinfo=None)
                            entry['timestamp'] = entry_timestamp
                            adsb_data.append(entry)
            except Exception as e:
                print(f"Error loading ADSB data for {blob.name}: {e}")

    total_hours = 24
    for hour in range(total_hours):
        prefix = f"ADS-B/{year}/{doy:03d}/{hour:02d}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        print(f"Processing hour {hour+1}/{total_hours}: {len(blobs)} files found")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(process_blob, blobs)

        print(f"Completed processing hour {hour+1}/{total_hours}")

    print(f"ADSB data loading complete. Total entries: {len(adsb_data)}")
    return adsb_data

def load_contrail_data(bucket_name, base_path, date):
    contrail_data = []
    storage_client = storage.Client()
    contrail_bucket = storage_client.bucket(bucket_name)
    year = date.year
    doy = date.timetuple().tm_yday

    prefix = f"{base_path}/{year}/{doy:03d}/"
    print(f"Searching for contrail data with prefix: {prefix}")
    blobs = list(contrail_bucket.list_blobs(prefix=prefix))
    
    if not blobs:
        print(f"No blobs found with prefix: {prefix}")
    
    for blob in blobs:
        if blob.name.endswith('positives_goes.json'):
            try:
                timestamp = extract_timestamp_from_path(blob.name)
                if timestamp:
                    data = json.loads(blob.download_as_text())
                    if data.get('d'):
                        for point in data['d']:
                            contrail_data.append({
                                'coordinates': point['c'],
                                'timestamp': datetime.fromtimestamp(point['t'], tz=timezone.utc)
                            })
                        print(f"Contrail data found for timestamp {timestamp}")
            except Exception as e:
                print(f"Error loading contrail data for blob {blob.name}: {e}")
    
    if not contrail_data:
        print(f"No contrail data found for the specified date.")
    else:
        print(f"Total contrail data points found: {len(contrail_data)}")
    
    return contrail_data

def main():
    contrail_bucket_name = "contrails_external"  # Replace with your contrail bucket name
    contrail_base_path = "tool_v2/detected_contrails_prod"  # Replace with your base path
    adsb_bucket_name = "contrailcast-trial-run-1-hierarchical"  # Replace with your ADSB bucket name
    min_altitude = 23000  # Set your desired minimum altitude here in feet

    date_str = input("Enter date (YYYY-MM-DD): ")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    # Load contrail data
    contrail_data = load_contrail_data(contrail_bucket_name, contrail_base_path, date)

    # Load ADSB data
    print(f"Loading ADSB data for date: {date_str}")
    adsb_data = load_adsb_data(adsb_bucket_name, date, min_altitude)
    print(f"ADSB data loaded successfully.")

    if not contrail_data and not adsb_data:
        print("No data found for the specified date.")
        return

    # Convert data to GeoDataFrames
    contrail_gdf = prepare_contrail_geodataframe(contrail_data)
    adsb_gdf = prepare_adsb_geodataframe(adsb_data)

    # Run Dash app
    run_dash_app(contrail_gdf, adsb_gdf)

def prepare_contrail_geodataframe(contrail_data):

    data = []
    for point in contrail_data:
        coords = [
            (point['coordinates'][1], point['coordinates'][0]),  # (lon1, lat1)
            (point['coordinates'][3], point['coordinates'][2])   # (lon2, lat2)
        ]
        geometry = LineString(coords)
        data.append({
            'geometry': geometry,
            'timestamp': point['timestamp'],
            'end_time': point['timestamp'] + timedelta(minutes=10)
        })

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf

def prepare_adsb_geodataframe(adsb_data):
    from shapely.geometry import Point
    import geopandas as gpd

    data = []
    for entry in adsb_data:
        geometry = Point(entry['lon'], entry['lat'])
        data.append({
            'geometry': geometry,
            'timestamp': entry['timestamp'],
            'end_time': entry['timestamp'] + timedelta(minutes=1),
            'altitude': entry['alt_geom'],
            'type': entry.get('t', None),
            'hex': entry.get('hex')
        })

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf

def run_dash_app(contrail_gdf, adsb_gdf):

    # Remove timezone information from timestamps
    contrail_gdf['timestamp'] = contrail_gdf['timestamp'].dt.tz_localize(None)
    contrail_gdf['end_time'] = contrail_gdf['end_time'].dt.tz_localize(None)
    adsb_gdf['timestamp'] = adsb_gdf['timestamp'].dt.tz_localize(None)
    adsb_gdf['end_time'] = adsb_gdf['end_time'].dt.tz_localize(None)

    # Combine timestamps from both datasets
    all_timestamps = pd.to_datetime(
        list(contrail_gdf['timestamp']) + list(adsb_gdf['timestamp'])
    ).sort_values().unique()
    start_time = all_timestamps.min()
    end_time = all_timestamps.max()
    time_range = pd.date_range(start=start_time, end=end_time, freq='1T')

    # Initialize Dash app
    app = dash.Dash(__name__)

    # server = app.server  # For deploying if needed
    contrail_layer = dl.LayerGroup(id='contrail-layer')
    adsb_layer = dl.LayerGroup(id='adsb-layer')

    app.layout = html.Div([
        html.H1("Contrail and ADSB Data Visualization"),
        html.Div([
            daq.BooleanSwitch(id='play-pause', on=False, label='Play/Pause', labelPosition='bottom'),
            dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),
            html.Div(id='current-time', style={'textAlign': 'center'}),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=len(time_range) - 1,
                value=0,
                marks={i: time_range[i].strftime('%H:%M') for i in range(0, len(time_range), 60)},
                step=1
            ),
        ], style={'width': '80%', 'margin': 'auto'}),
        dl.Map(
            id='map',
            center=[39.8283, -98.5795],  # Center of the US
            zoom=4,
            style={'width': '100%', 'height': '80vh'},
            children=[
                dl.TileLayer(),
                contrail_layer,
                adsb_layer,
                dl.LayersControl(
                    [dl.Overlay(contrail_layer, name="Contrails", checked=True),
                    dl.Overlay(adsb_layer, name="ADSB Data", checked=True)]
                ),
                dl.FullScreenControl(),
            ]
        )
    ])

    @app.callback(
        Output('interval-component', 'disabled'),
        Input('play-pause', 'on')
    )
    def toggle_interval(on):
        return not on  # Interval is disabled when 'on' is False

    @app.callback(
        Output('time-slider', 'value'),
        Input('interval-component', 'n_intervals'),
        State('time-slider', 'value'),
        State('time-slider', 'max')
    )
    def update_slider(n_intervals, current_value, max_value):
        if current_value < max_value:
            return current_value + 1
        else:
            return 0  # Loop back to the start

    @app.callback(
        Output('current-time', 'children'),
        Input('time-slider', 'value')
    )
    def update_current_time(slider_value):
        current_time = time_range[slider_value]
        return f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    class TimestampEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            return super().default(obj)

    @app.callback(
        [Output('contrail-layer', 'children'),
        Output('adsb-layer', 'children')],
        [Input('time-slider', 'value')]
    )
    def update_layers(selected_time_index):
        try:
            selected_time = time_range[selected_time_index].replace(tzinfo=None)
            print(f"\nSelected time: {selected_time}")

            # Filter contrail data
            contrail_current = contrail_gdf[
                (contrail_gdf['timestamp'] <= selected_time) &
                (contrail_gdf['end_time'] > selected_time)
            ].copy()

            # Filter ADSB data for the last up to 10 minutes
            adsb_current = adsb_gdf[
                (adsb_gdf['timestamp'] <= selected_time) &
                (adsb_gdf['timestamp'] >= selected_time - timedelta(minutes=10))
            ].copy()

            # Prepare trajectory and current position data
            trajectories = []
            positions = []
            if not adsb_current.empty:
                # Group ADSB data by aircraft identifier
                adsb_grouped = adsb_current.groupby('hex')

                for hex_code, group in adsb_grouped:
                    group = group.sort_values('timestamp')
                    if len(group) >= 2:
                        # Create trajectory LineString
                        trajectory = LineString(group.geometry.tolist())
                        trajectories.append({
                            'hex': hex_code,
                            'geometry': trajectory,
                            'timestamp': selected_time
                        })
                    # Get current position (last point)
                    current_position = group.iloc[-1]
                    positions.append({
                        'hex': hex_code,
                        'geometry': current_position.geometry,
                        'timestamp': current_position.timestamp,
                        'altitude': current_position.altitude
                    })

            # Create GeoDataFrames
            trajectory_gdf = gpd.GeoDataFrame(
                trajectories,
                columns=['hex', 'geometry', 'timestamp'],
                geometry='geometry',
                crs="EPSG:4326"
            )

            positions_gdf = gpd.GeoDataFrame(
                positions,
                columns=['hex', 'geometry', 'timestamp', 'altitude'],
                geometry='geometry',
                crs="EPSG:4326"
            )

            # Contrail Layer
            contrail_geojson = []
            if not contrail_current.empty:
                contrail_current['timestamp'] = contrail_current['timestamp'].astype(str)
                contrail_current['end_time'] = contrail_current['end_time'].astype(str)
                contrail_geojson = [dl.GeoJSON(
                    data=contrail_current.to_crs(epsg=4326).__geo_interface__,
                    style={'color': 'red', 'weight': 2, 'opacity': 0.8},
                    hoverStyle={'color': 'yellow', 'weight': 5, 'opacity': 1},
                    zoomToBoundsOnClick=True
                )]

            # Prepare ADSB Layer Children
            adsb_layer_children = []

            # Add Trajectories to ADSB Layer
            if not trajectory_gdf.empty:
                adsb_trajectory_geojson = dl.GeoJSON(
                    data=trajectory_gdf.__geo_interface__,
                    style={'color': 'blue', 'weight': 2, 'opacity': 0.8},
                    hoverStyle={'color': 'yellow', 'weight': 5, 'opacity': 1},
                    zoomToBoundsOnClick=True
                )
                adsb_layer_children.append(adsb_trajectory_geojson)

            # Add Altitude Markers to ADSB Layer using DivMarker
            if not positions_gdf.empty:
                for idx, row in positions_gdf.iterrows():
                    lat = row.geometry.y
                    lon = row.geometry.x
                    altitude = row.altitude

                    # Create a DivMarker to display the altitude as text
                    altitude_marker = dl.DivMarker(
                        position=[lat, lon],
                        iconOptions={
                            'html': f'<div style="font-size: 12px; color: blue; text-align: center;">{altitude} ft</div>',
                            'iconSize': [50, 20],  # Adjust as needed
                            'iconAnchor': [25, 10],  # Center the icon
                            'className': ''  # Optional CSS class
                        }
                    )
                    adsb_layer_children.append(altitude_marker)

            return contrail_geojson, adsb_layer_children

        except Exception as e:
            print(f"Exception in update_layers: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []
            
    # Run the Dash app
    app.run_server(debug=True, use_reloader=False)  # Disable reloader to prevent duplicate execution

if __name__ == "__main__":
    main()