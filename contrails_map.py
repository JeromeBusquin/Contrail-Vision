import json
from google.cloud import storage
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta, timezone
import os
import re
from google.api_core import retry
import concurrent.futures


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
    adsb_data = {}
    year = date.year
    doy = date.timetuple().tm_yday
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def process_blob(blob):
        if blob.name.endswith('adsb.json'):
            timestamp = extract_timestamp_from_path(blob.name)
            if timestamp:
                try:
                    data = json.loads(blob.download_as_text())
                    filtered_data = [
                        entry for entry in data 
                        if 'alt_geom' in entry and entry['alt_geom'] is not None and entry['alt_geom'] >= min_altitude
                    ]
                    if filtered_data:
                        return timestamp, filtered_data
                except Exception as e:
                    print(f"Error loading ADSB data for {blob.name}: {e}")
        return None, None

    total_hours = 24
    for hour in range(total_hours):
        prefix = f"ADS-B/{year}/{doy:03d}/{hour:02d}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        print(f"Processing hour {hour+1}/{total_hours}: {len(blobs)} files found")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_blob = {executor.submit(process_blob, blob): blob for blob in blobs}
            for future in concurrent.futures.as_completed(future_to_blob):
                timestamp, filtered_data = future.result()
                if timestamp and filtered_data:
                    adsb_data[timestamp] = filtered_data
        
        print(f"Completed processing hour {hour+1}/{total_hours}")

    print(f"ADSB data loading complete. Total timestamps with data: {len(adsb_data)}")
    return adsb_data

def create_map_with_data(contrail_data, adsb_data):
    if not contrail_data and not adsb_data:
        m = folium.Map(location=[0, 0], zoom_start=2)
        folium.Marker(
            [0, 0],
            popup="No data found for the specified date",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        return m

    all_lats = [coord for timestamp_data in contrail_data.values() for point in timestamp_data['d'] for coord in [point['c'][0], point['c'][2]]]
    all_lats.extend([entry['lat'] for data in adsb_data.values() for entry in data])
    all_lons = [coord for timestamp_data in contrail_data.values() for point in timestamp_data['d'] for coord in [point['c'][1], point['c'][3]]]
    all_lons.extend([entry['lon'] for data in adsb_data.values() for entry in data])
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    # Add contrail data
    contrail_features = []
    for timestamp, timestamp_data in contrail_data.items():
        end_time = timestamp + timedelta(minutes=10)
        for point in timestamp_data['d']:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[point['c'][1], point['c'][0]], [point['c'][3], point['c'][2]]]
                },
                'properties': {
                    'times': [timestamp.isoformat(), end_time.isoformat()],
                    'style': {
                        'color': 'red',
                        'weight': 2,
                        'opacity': 0.8
                    }
                }
            }
            contrail_features.append(feature)

    # Add ADSB data
    adsb_features = []
    for timestamp, entries in adsb_data.items():
        for entry in entries:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [entry['lon'], entry['lat']]
                },
                'properties': {
                    'times': [timestamp.isoformat(), (timestamp + timedelta(minutes=1)).isoformat()],
                    'style': {
                        'color': 'blue',
                        'fillColor': 'blue',
                        'fillOpacity': 0.8,
                        'weight': 2,
                        'radius': 5
                    },
                    'popup': f"Type: {entry['t']}, Altitude: {entry['alt_geom']}ft"
                }
            }
            adsb_features.append(feature)

    # Combine features
    all_features = contrail_features + adsb_features

    # Create a single TimestampedGeoJson layer
    timestamped_layer = TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': all_features},
        period='PT1M',
        duration=None,
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True,
    )

    timestamped_layer.add_to(m)

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

def main():
    contrail_bucket_name = "contrails_external"
    contrail_base_path = "tool_v2/detected_contrails_prod"
    adsb_bucket_name = "contrailcast-trial-run-1-hierarchical"
    min_altitude = 10000  # Set your desired minimum altitude here
    
    while True:
        date_str = input("Enter date (YYYY-MM-DD): ")
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            year, doy = date.year, date.timetuple().tm_yday
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Load contrail data
    contrail_data = {}
    storage_client = storage.Client()
    contrail_bucket = storage_client.bucket(contrail_bucket_name)
    
    prefix = f"{contrail_base_path}/{year}/{doy:03d}/"
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
                        contrail_data[timestamp] = data
                        print(f"Contrail data found for timestamp {timestamp}")
            except Exception as e:
                print(f"Error loading contrail data for blob {blob.name}: {e}")

    if not contrail_data:
        print(f"No contrail data found for the specified date: {date_str}")
    else:
        print(f"Total contrail data points found: {len(contrail_data)}")

    # Load ADSB data
    print(f"Loading ADSB data for date: {date_str}")
    adsb_data = load_adsb_data(adsb_bucket_name, date, min_altitude)
    print(f"ADSB data loaded successfully. Total minutes with data: {len(adsb_data)}")

    if not contrail_data and not adsb_data:
        print("No data found for the specified date.")
        return

    m = create_map_with_data(contrail_data, adsb_data)

    output_file = f"combined_map_{date_str}.html"
    m.save(output_file)
    print(f"Map saved as {output_file}")

if __name__ == "__main__":
    main()