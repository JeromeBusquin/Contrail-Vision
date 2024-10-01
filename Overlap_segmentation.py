import sys
import os
import numpy as np
import torch
import cv2
import requests
from yolox.tracker.byte_tracker import BYTETracker

# Define the Args class for BYTETracker
class Args:
    def __init__(self):
        self.track_thresh = 0.2  # Tracking confidence threshold
        self.track_buffer = 600  # Buffer size to maintain IDs across frames
        self.match_thresh = 0.3  # Lower IoU threshold to better handle overlaps
        self.mot20 = False

# Function to calculate the minimum area rotated bounding box
def get_rotated_bbox(points):
    rect = cv2.minAreaRect(points)  # Find the minimum area rectangle
    box = cv2.boxPoints(rect)  # Get the four corner points of the rotated rectangle
    box = np.int0(box)  # Convert box points to integers
    return box, rect

# Function to perform inference using the Hosted Inference API
def infer_image(frame_rgb):
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
    api_key = '6Kzy0L8WP8idrygkLOUr'  # Replace with your API key
    project_name = 'contrails-0lwhn-ibga0'  # Your project name
    version_number = '2'  # Your model version number
    url = (
        f'https://detect.roboflow.com/{project_name}/{version_number}'
        f'?api_key={api_key}&confidence=30&overlap=90'
    )

    # Send the POST request
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Failed to get inference: {response.status_code}, {response.text}")
        return None

# Function to process a single frame and track the objects
def process_frame(frame_index, frame_predictions, byte_tracker, frame):
    current_points = []
    masks = []

    # Set the confidence threshold
    confidence_threshold = 0.2

    # Filter predictions based on confidence threshold
    predictions = [
        pred for pred in frame_predictions['predictions']
        if pred.get('confidence', 1.0) >= confidence_threshold
    ]

    # Inside the process_frame function, modify the loop over predictions
    for prediction in predictions:
        if 'points' in prediction and prediction['points']:
             # Extract points
            points = np.array(
                [[point['x'], point['y']] for point in prediction['points']],
                dtype=np.int32
            )

            # Check if there are at least 3 points
            if len(points) >= 3:
                 # Get the rotated bounding box for tracking
                rotated_box, rect = get_rotated_bbox(points)
                current_points.append((rotated_box, rect))

                # Store the segmentation mask for the output
                masks.append(points)
            else:
                print(f"Skipping prediction with insufficient points: {points}")
        else:
            print(f"Skipping prediction with no 'points' or empty 'points': {prediction}")


    # Check if there are any detections to track
    if len(current_points) == 0:
        print(f"Frame {frame_index}: No detections.")
        return None

    # Convert detections into the ByteTrack format (bounding boxes)
    detections = np.array([
        [
            rect[0][0] - rect[1][0]/2,
            rect[0][1] - rect[1][1]/2,
            rect[0][0] + rect[1][0]/2,
            rect[0][1] + rect[1][1]/2,
            1.0
        ]
        for _, rect in current_points
    ])

    # Convert to a PyTorch tensor
    detections_tensor = torch.from_numpy(detections).float()

    # Image size
    img_size = frame.shape[:2]  # (height, width)
    img_info = (img_size[0], img_size[1])  # Image info in (height, width)

    # Track objects using ByteTrack
    tracked_objects = byte_tracker.update(detections_tensor, img_info, img_size)

    # Visualize and print tracked objects
    if len(tracked_objects) > 0:
        print(f"Frame {frame_index}: Tracked {len(tracked_objects)} objects.")
        for obj in tracked_objects:
            print(f"Object {obj.track_id} with bounding box: {obj.tlbr}")

            # Draw the rotated bounding box
            for (rotated_box, _) in current_points:
                cv2.polylines(frame, [rotated_box], True, (0, 255, 0), 2)  # Green bounding box

            # Apply masks on the frame
            for mask_points in masks:
                cv2.fillPoly(frame, [mask_points], (255, 0, 0))  # Blue mask overlay

            # Add the track ID to the object
            cv2.putText(
                frame, f"ID: {obj.track_id}",
                (int(obj.tlbr[0]), int(obj.tlbr[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
    else:
        print(f"Frame {frame_index}: No objects tracked.")

    return tracked_objects

# Main function to handle video prediction, tracking, and output video generation
def main():
    print("Current working directory:", os.getcwd())

    video_path = 'C:/Users/jerom/Projects/Input_video/US9999_20231122_155519_UC_US9999_20231122_155519_adsb_timelapse.mp4'

    # Load the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output video writer
    output_video_path = 'C:/Users/jerom/Projects/RMS_data/InferenceTest/output_with_tracking.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


    # Initialize ByteTrack
    byte_tracker = BYTETracker(Args())


    frame_index = 0
    # Process each frame and perform tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get predictions for the current frame
        frame_predictions = infer_image(frame_rgb)
        if frame_predictions is None:
            print(f"Frame {frame_index}: Inference failed.")
            continue

        # Process the frame for tracking and overlay masks and bounding boxes
        tracked_objects = process_frame(frame_index, frame_predictions, byte_tracker, frame)

        if tracked_objects is not None:
            print(f"Tracked {len(tracked_objects)} objects in this frame.")

        # Write the frame with masks and tracking info to the output video
        out.write(frame)
        frame_index += 1

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print completion message
    print("Video processing complete! The output video is saved at: ", output_video_path)

if __name__ == "__main__":
    main()
