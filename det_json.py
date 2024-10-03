import os
import sys
import json
import numpy as np
import cv2
from google.cloud import storage
from datetime import datetime
import requests

sys.path.append('/Users/jerom/Projects/RMS-Contrail')
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
    confidence_threshold = 0.2

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
        if len(mask_points) >= 2:  # Ensure there are enough points to form a line
            start_x, start_y = mask_points[0]
            end_x, end_y = mask_points[-1]
            start_lat, start_lon = XyHt2Geo(platepar, start_x + roi_x, start_y + roi_y, 11000)
            end_lat, end_lon = XyHt2Geo(platepar, end_x + roi_x, end_y + roi_y, 11000)
            lat_lon_data.append([start_lat, start_lon, end_lat, end_lon])

    return lat_lon_data

# Function to read the platepar file from GCS directly and load it into Platepar
def load_platepar_from_gcs(bucket_name, platepar_blob_name):
    # Initialize GCS client
    storage_client = storage.Client()

    # Get the platepar blob from GCS
    platepar_blob = storage_client.bucket(bucket_name).blob(platepar_blob_name)

    # Download the platepar data as bytes
    platepar_data = platepar_blob.download_as_bytes()

    # Initialize the Platepar object
    platepar = Platepar()

    # Read the platepar data from bytes
    # Assuming Platepar can read from a string or bytes-like object
    platepar.read(platepar_data)

    return platepar


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

# Main function to handle frame processing and saving lat-long data
def main():
    # Set up GCS bucket and API details
    input_bucket_name = "contrailcast-trial-run-1-hierarchical"
    input_folder_prefix = "US9999_20231122_155519/"

    output_bucket_name = "contrailcast-trial-run-1-hierarchical"
    output_folder_prefix = "Ground_Detected/"

    api_key = '6Kzy0L8WP8idrygkLOUr'  # Replace with your actual API key
    project_name = 'contrails-0lwhn-ibga0'  # Replace with your project name
    version_number = '2'  # Replace with your model version number

    # Flag to process every n-th frame
    process_every_nth_frame = True
    n = 120  # Process every 5th frame

    # Initialize GCS client
    storage_client = storage.Client()

    # Download the platepar file from GCS
    platepar_blob_name = input_folder_prefix + 'platepar_cmn2011.cal'

    # Load the platepar directly from GCS
    platepar = load_platepar_from_gcs(input_bucket_name, platepar_blob_name)

    # List all image blobs in the input bucket
    blobs = list(storage_client.list_blobs(input_bucket_name, prefix=input_folder_prefix))

    # Filter and sort blobs to ensure correct order
    image_blobs = [
        blob for blob in blobs
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_blobs.sort(key=lambda x: x.name)

    # Limit the blobs to every n-th frame if the flag is set
    if process_every_nth_frame:
        image_blobs = image_blobs[::n]

    # Initialize an empty list to store the contrail detections in the desired format
    contrail_detections = []

    # Process each frame
    for idx, blob in enumerate(image_blobs):
        # Extract the filename from the blob name
        filename = os.path.basename(blob.name)

        # Example filename: UC_US9999_20231122_155519_480.jpg
        # Split the filename to extract date and time
        parts = filename.split('_')
        if len(parts) < 5:
            print(f"Unexpected filename format: {filename}")
            continue

        date_str = parts[2]  # '20231122'
        time_str = parts[3]  # '155519'

        datetime_str = date_str + time_str  # '20231122155519'

        try:
            timestamp_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
            frame_timestamp = int(timestamp_dt.timestamp())
        except ValueError as e:
            print(f"Error parsing date and time from filename {filename}: {e}")
            continue

        print(f"Processing frame {filename} with timestamp {timestamp_dt} (Frame {idx+1}/{len(image_blobs)})")

        # Download image data into memory
        image_data = blob.download_as_bytes()
        image_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"Failed to read image {filename}")
            continue

        # Define ROI parameters (update as needed)
        roi_x, roi_y, roi_w, roi_h = 0, 0, frame.shape[1], frame.shape[0]

        # Crop frame to ROI if needed
        roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        # Get predictions for the current frame
        frame_predictions = infer_image(frame_rgb, api_key, project_name, version_number)
        if frame_predictions is None:
            print(f"Frame {filename}: Inference failed.")
            continue

        # Process the frame and get lat-long data for masks
        lat_lon_data = process_frame(frame_predictions, platepar, roi_x, roi_y)

        # Store the detections in the desired format
        for lat_lon in lat_lon_data:
            contrail_detections.append({
                "c": lat_lon,  # Coordinates of the contrail
                "t": frame_timestamp  # Timestamp in Unix format for this frame
            })

    # Save the detections as a JSON file to GCS in the desired format
    json_data = {"d": contrail_detections}
    output_file_name = f"{output_folder_prefix}contrail_detections_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    save_to_gcs(output_bucket_name, output_file_name, json_data)

    print("Processing complete. JSON file with detections saved to GCS.")

if __name__ == "__main__":
    main()
