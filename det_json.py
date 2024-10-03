import os
import tempfile
import sys
import json
import numpy as np
import cv2
from google.cloud import storage
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level
projects_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Append the RMS-Contrail path
rms_contrail_path = os.path.join(projects_dir, 'RMS-Contrail')
sys.path.append(rms_contrail_path)

from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
from RMS.Formats.Platepar import Platepar



# Function to perform inference using the Hosted Inference API
def infer_image(frame_rgb, api_key, project_name, version_number):
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
        f'?api_key={api_key}&confidence=30&overlap=90'
    )

    # Send the POST request
    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get inference: {response.status_code}, {response.text}")
        return None

# Function to process a single frame and get lat-long data
def process_frame(frame_predictions, platepar, roi_x, roi_y):
    masks = []
    # Set the confidence threshold
    confidence_threshold = 0.05
    # Filter predictions based on confidence threshold
    predictions = [
        pred for pred in frame_predictions['predictions']
        if pred.get('confidence', 1.0) >= confidence_threshold
    ]
    # Iterate over the segmentation data
    for prediction in predictions:
        if 'points' in prediction and prediction['points']:
            # Extract points
            points = np.array(
                [[point['x'], point['y']] for point in prediction['points']],
                dtype=np.int32
            )
            # Check if there are at least 3 points
            if len(points) >= 3:
                masks.append(points)
            else:
                print(f"Skipping prediction with insufficient points: {points}")

    # Convert masks to lat-long pairs
    lat_lon_data = []
    for mask_points in masks:
        polygon_lat_lon = []
        for point in mask_points:
            x, y = point
            lat, lon = XyHt2Geo(platepar, x + roi_x, y + roi_y, 11000)
            polygon_lat_lon.append([lat, lon])
        lat_lon_data.append(polygon_lat_lon)

    return lat_lon_data


def load_platepar_from_gcs(bucket_name, platepar_blob_name):
    # Initialize GCS client
    storage_client = storage.Client()
    
    # Get the platepar blob from GCS
    bucket = storage_client.bucket(bucket_name)
    platepar_blob = bucket.blob(platepar_blob_name)
    
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


# Function to save data to GCS as a JSON file
def save_to_gcs(bucket_name, file_name, data):
    # Initialize GCS client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob and upload the data
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(data), content_type='application/json')

    print(f"File {file_name} uploaded to {bucket_name}.")

def process_frame_parallel(args):
    blob, platepar, roi_x, roi_y, api_key, project_name, version_number = args
    filename = os.path.basename(blob.name)
    
    # Extract timestamp (same as before)
    parts = filename.split('_')
    if len(parts) < 5:
        print(f"Unexpected filename format: {filename}")
        return None

    date_str, time_str = parts[2], parts[3]
    datetime_str = date_str + time_str

    try:
        timestamp_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        frame_timestamp = int(timestamp_dt.timestamp())
    except ValueError as e:
        print(f"Error parsing date and time from filename {filename}: {e}")
        return None

    print(f"Processing frame {filename} with timestamp {timestamp_dt}")

    # Download and process image (same as before)
    image_data = blob.download_as_bytes()
    image_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"Failed to read image {filename}")
        return None

    roi_frame = frame[roi_y:roi_y + frame.shape[0], roi_x:roi_x + frame.shape[1]]
    frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

    # Get predictions and process frame
    frame_predictions = infer_image(frame_rgb, api_key, project_name, version_number)
    if frame_predictions is None:
        print(f"Frame {filename}: Inference failed.")
        return None

    lat_lon_data = process_frame(frame_predictions, platepar, roi_x, roi_y)

    # Return detections for this frame
    return [{"c": lat_lon, "t": frame_timestamp} for lat_lon in lat_lon_data]

# Main function to handle frame processing and saving lat-long data
def main():
    # Set up GCS bucket and API details
    input_bucket_name = "contrailcast-trial-run-1-hierarchical"
    input_folder_prefix = "US9999_20231122_155519/"

    output_bucket_name = "contrailcast-trial-run-1-hierarchical"
    output_folder_prefix = "detected_contrails_ground/"

    api_key = '6Kzy0L8WP8idrygkLOUr'  # Replace with your actual API key
    project_name = 'contrails-0lwhn-ibga0'  # Replace with your project name
    version_number = '2'  # Replace with your model version number

    # Flag to process every n-th frame
    process_every_nth_frame = True
    n = 12

    # Initialize GCS client
    storage_client = storage.Client()

    # Download the platepar file from GCS
    platepar_blob_name = input_folder_prefix + 'platepar_cmn2011.cal'

    # Load the platepar directly from GCS
    platepar = load_platepar_from_gcs(input_bucket_name, platepar_blob_name)

    # List and filter blobs (same as before)
    blobs = list(storage_client.list_blobs(input_bucket_name, prefix=input_folder_prefix))
    image_blobs = [
        blob for blob in blobs
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_blobs.sort(key=lambda x: x.name)

    if process_every_nth_frame:
        image_blobs = image_blobs[::n]

    # Prepare arguments for parallel processing
    process_args = [
        (blob, platepar, 0, 0, api_key, project_name, version_number)
        for blob in image_blobs
    ]

    # Use ThreadPoolExecutor for I/O-bound operations (network requests)
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        future_to_frame = {executor.submit(process_frame_parallel, args): args for args in process_args}
        
        contrail_detections = []
        for future in as_completed(future_to_frame):
            result = future.result()
            if result:
                contrail_detections.extend(result)

    # Save the detections as a JSON file to GCS
    json_data = {"d": contrail_detections}
    output_file_name = f"{output_folder_prefix}contrail_detections_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    save_to_gcs(output_bucket_name, output_file_name, json_data)

    print("Processing complete. JSON file with detections saved to GCS.")

if __name__ == "__main__":
    main()