import json
from google.cloud import storage
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import os
import re

def list_blobs_with_prefix(bucket_name, prefix):
    storage_client = storage.Client()
    return storage_client.list_blobs(bucket_name, prefix=prefix)

def load_json_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    json_data = blob.download_as_text()
    return json.loads(json_data)

def extract_timestamp_from_path(path):
    match = re.search(r'/(\d{4})/(\d{3})/\d{2}/(\d+)/', path)
    if match:
        year, doy, unix_timestamp = match.groups()
        try:
            timestamp = int(unix_timestamp)
            return datetime.utcfromtimestamp(timestamp)
        except ValueError as e:
            print(f"Error converting timestamp: {e}")
    return None

def create_map_with_timestamped_geojson(data):
    if not data:
        m = folium.Map(location=[0, 0], zoom_start=2)
        folium.Marker(
            [0, 0],
            popup="No contrail data found for the specified date",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        return m

    all_lats = [coord for timestamp_data in data.values() for point in timestamp_data['d'] for coord in [point['c'][0], point['c'][2]]]
    all_lons = [coord for timestamp_data in data.values() for point in timestamp_data['d'] for coord in [point['c'][1], point['c'][3]]]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    features = []
    for timestamp, timestamp_data in data.items():
        for point in timestamp_data['d']:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[point['c'][1], point['c'][0]], [point['c'][3], point['c'][2]]]
                },
                'properties': {
                    'times': [timestamp.isoformat(), (timestamp + timedelta(minutes=10)).isoformat()],
                    'style': {
                        'color': 'red',
                        'weight': 2,
                        'opacity': 0.8
                    }
                }
            }
            features.append(feature)

    timestamped_geojson = TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='PT10M',
        duration='PT10M',
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True,
    )

    timestamped_geojson.add_to(m)

    custom_css = """
    <style>
    .leaflet-bottom.leaflet-left {
        width: 100%;
    }
    .leaflet-control-container .leaflet-timeline-controls {
        box-sizing: border-box;
        width: 100%;
        margin: 0;
        margin-bottom: 15px;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(custom_css))

    return m

def date_to_year_doy(date):
    year = date.year
    doy = date.timetuple().tm_yday
    return year, f"{doy:03d}"

def main():
    bucket_name = "contrails_external"
    base_path = "tool_v2/detected_contrails_prod"
    
    while True:
        date_str = input("Enter date (YYYY-MM-DD): ")
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            year, doy = date_to_year_doy(date)
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    all_data = {}
    data_found = False

    prefix = f"{base_path}/{year}/{doy}/"
    print(f"Searching for blobs with prefix: {prefix}")
    blobs = list_blobs_with_prefix(bucket_name, prefix)
    for blob in blobs:
        if blob.name.endswith('positives_goes.json'):
            try:
                timestamp = extract_timestamp_from_path(blob.name)
                if timestamp:
                    data = load_json_from_gcs(bucket_name, blob.name)
                    if data['d']:
                        all_data[timestamp] = data
                        data_found = True
                        print(f"Data found for timestamp {timestamp}")
            except Exception as e:
                print(f"Error loading data for blob {blob.name}: {e}")

    if not data_found:
        print(f"No data found for the specified date: {date_str}")
    else:
        print(f"Total data points found: {len(all_data)}")

    m = create_map_with_timestamped_geojson(all_data)

    output_file = f"contrails_map_{date_str}_timestamped.html"
    m.save(output_file)
    print(f"Map saved as {output_file}")

if __name__ == "__main__":
    main()