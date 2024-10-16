import os
import tempfile
import sys
import json
import numpy as np
import cv2
from google.cloud import storage
from datetime import datetime, timezone
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import re

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level
projects_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Append the RMS-Contrail path
rms_contrail_path = os.path.join(projects_dir, 'RMS-Contrail')
sys.path.append(rms_contrail_path)

from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
from RMS.Formats.Platepar import Platepar

# User inputs
TARGET_YEAR = "2023"
TARGET_MONTH = "11"

# GCS bucket information
BUCKET_NAME = "contrailcast-trial-run-1-hierarchical"
FOLDER_PATTERN = re.compile(r"(US\d+_\d{8}_\d+)/Frames")

# Initialize GCS client
client = storage.Client()

# List blobs and filter based on user input
def list_target_folders(bucket_name, year, month):
    bucket = client.get_bucket(bucket_name)
    target_folders = set()
    blobs = bucket.list_blobs(prefix="US", delimiter="/")
    for page in blobs.pages:
        for folder_prefix in page.prefixes:
            match = re.match(r"(US\d+_\d{8}_\d+)/", folder_prefix)
            if match:
                folder_name = match.group(1)
                # Extract year and month from the folder name
                date_str = folder_name.split('_')[1]  # e.g., "20231011"
                blob_year, blob_month = date_str[:4], date_str[4:6]
                if blob_year == year and blob_month == month:
                    target_folders.add(folder_name)
    return list(target_folders)

# Function to load the Platepar file from GCS
def load_platepar_from_gcs(bucket_name, platepar_blob_name):
    # Initialize GCS client
    storage_client = storage.Client()
    # Get the platepar blob from GCS
    bucket = storage_client.bucket(bucket_name)
    platepar_blob = bucket.blob(platepar_blob_name)
    if not platepar_blob.exists():
        print(f"Platepar file {platepar_blob_name} not found in bucket {bucket_name}")
        return None
    # Download the content directly into memory
    platepar_content = platepar_blob.download_as_text()
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(platepar_content)
        temp_file_path = temp_file.name
    try:
        # Initialize the Platepar object
        platepar = Platepar()
        # Read the platepar data from the temporary file
        platepar.read(temp_file_path)
        return platepar
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

# Function to perform inference using the Hosted Inference API
def infer_image(frame_rgb, api_key, project_name, version_number, max_retries=3, retry_delay=5):
    # Prepare the image for sending
    retval, buffer = cv2.imencode('.jpg', frame_rgb)
    if not retval:
        print("Failed to encode image")
        return None

    # Prepare the multipart/form-data
    files = {
        'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')
    }

    # Build the URL with confidence and overlap parameters
    url = (
        f'https://detect.roboflow.com/{project_name}/{version_number}'
        f'?api_key={api_key}&confidence=20&overlap=70'
    )

    for attempt in range(max_retries):
        try:
            # Send the POST request
            response = requests.post(url, files=files, timeout=30)  # Add a timeout
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Attempt {attempt + 1}: Failed to get inference: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Network error occurred: {e}")

        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    print("Max retries reached. Inference failed.")
    return None

# Function to process a single frame and get lat-long data
def process_frame(frame_predictions, platepar, roi_x, roi_y):
    lat_lon_data = []
    confidence_threshold = 0.05
    predictions = [
        pred for pred in frame_predictions['predictions']
        if pred.get('confidence', 0) >= confidence_threshold
    ]
    for prediction in predictions:
        if 'points' in prediction and prediction['points']:
            points = np.array(
                [[point['x'], point['y']] for point in prediction['points']],
                dtype=np.float32
            )
            polygon_lat_lon = []
            for point in points:
                x, y = point
                lat, lon = XyHt2Geo(platepar, x + roi_x, y + roi_y, 10000)
                polygon_lat_lon.append([lat, lon])
            if polygon_lat_lon and polygon_lat_lon[0] != polygon_lat_lon[-1]:
                polygon_lat_lon.append(polygon_lat_lon[0])
            if polygon_lat_lon:
                lat_lon_data.append(polygon_lat_lon)
    return lat_lon_data

# Function to save data to GCS as a JSON file
def get_year_doy_hour(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.year, dt.timetuple().tm_yday, dt.hour

def save_to_gcs_organized(bucket_name, folder_prefix, data):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    organized_data = {}
    for detection in data["d"]:
        timestamp = detection["t"]
        year, doy, hour = get_year_doy_hour(timestamp)
        key = (year, doy, hour)
        if key not in organized_data:
            organized_data[key] = {"d": []}
        organized_data[key]["d"].append(detection)
    for (year, doy, hour), hourly_data in organized_data.items():
        file_path = f"{folder_prefix}{year:04d}/{doy:03d}/{hour:02d}/positives_ground.json"
        blob = bucket.blob(file_path)
        blob.upload_from_string(json.dumps(hourly_data), content_type='application/json')
        print(f"File {file_path} uploaded to {bucket_name}.")

# Function to process frames in parallel
def process_frame_parallel(args):
    blob, platepar, roi_x, roi_y, api_key, project_name, version_number = args
    filename = os.path.basename(blob.name)
    parts = filename.split('_')
    if len(parts) < 5:
        print(f"Unexpected filename format: {filename}")
        return None
    date_str, time_str = parts[2], parts[3]
    datetime_str = date_str + time_str
    try:
        timestamp_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        frame_timestamp = int(timestamp_dt.timestamp())
    except ValueError as e:
        print(f"Error parsing date and time from filename {filename}: {e}")
        return None
    print(f"Processing frame {filename} with timestamp {timestamp_dt}")
    image_data = blob.download_as_bytes()
    image_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Failed to read image {filename}")
        return None
    roi_frame = frame[roi_y:roi_y + frame.shape[0], roi_x:roi_x + frame.shape[1]]
    frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    frame_predictions = infer_image(frame_rgb, api_key, project_name, version_number)
    if frame_predictions is None:
        print(f"Frame {filename}: Inference failed.")
        return None
    lat_lon_data = process_frame(frame_predictions, platepar, roi_x, roi_y)
    return [{"c": lat_lon, "t": frame_timestamp} for lat_lon in lat_lon_data]

# Main logic to process all target folders
def main():
    input_bucket_name = "contrailcast-trial-run-1-hierarchical"
    output_bucket_name = "contrailcast-trial-run-1-hierarchical"
    output_folder_prefix = "detected_contrails_ground/"
    api_key = '6Kzy0L8WP8idrygkLOUr'  # Replace with your actual API key
    project_name = 'contrails-0lwhn-ibga0'  # Replace with your project name
    version_number = '2'  # Replace with your model version number
    process_every_nth_frame = True
    n = 12
    target_folders = list_target_folders(BUCKET_NAME, TARGET_YEAR, TARGET_MONTH)
    if not target_folders:
        print(f"No folders found for year {TARGET_YEAR} and month {TARGET_MONTH}")
    else:
        for folder in target_folders:
            platepar_blob_name = f"{folder}/platepar_cmn2011.cal"
            platepar = load_platepar_from_gcs(input_bucket_name, platepar_blob_name)
            if platepar is None:
                continue
            blobs = list(client.list_blobs(input_bucket_name, prefix=f"{folder}/Frames/"))
            image_blobs = [
                blob for blob in blobs
                if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            image_blobs.sort(key=lambda x: x.name)
            if process_every_nth_frame:
                image_blobs = image_blobs[::n]
            process_args = [
                (blob, platepar, 0, 0, api_key, project_name, version_number)
                for blob in image_blobs
            ]
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
                future_to_frame = {executor.submit(process_frame_parallel, args): args for args in process_args}
                contrail_detections = []
                for future in as_completed(future_to_frame):
                    result = future.result()
                    if result:
                        contrail_detections.extend(result)
                json_data = {"d": contrail_detections}
                print(f"Saving: {output_bucket_name}, {output_folder_prefix}")
                save_to_gcs_organized(output_bucket_name, output_folder_prefix, json_data)
                print("Processing complete. JSON file with detections saved to GCS.")

if __name__ == "__main__":
    main()