import numpy as np
from roboflow import Roboflow
import cv2

# Initialize Roboflow and the model as you have before
rf = Roboflow(api_key="6Kzy0L8WP8idrygkLOUr")
workspace = rf.workspace("contrailcast")
project = workspace.project("contrails-0lwhn-ibga0")
model = project.version(1).model

# Path to your input and output videos
video_path = 'C:/Users/jerom/Projects/Input_video/ground_sat_match.mp4'
output_path = 'C:/Users/jerom/Projects/RMS_data/InferenceTest/annotated_video_gts.mp4'

# Run prediction on the video
job_id, signed_url, expire_time = model.predict_video(
    video_path,
    fps=30,
    prediction_type="batch-video",
)
results = model.poll_until_video_results(job_id)

print("Video predictions completed.")

def callback(scene: np.ndarray, index: int) -> np.ndarray:
    # Get predictions for the current frame
    frame_predictions = results['contrails-0lwhn-ibga0'][index]
    
    # Count the number of objects detected
    num_objects = len(frame_predictions['predictions'])
    print(f"Frame {index + 1}: Detected {num_objects} object(s)")

    # Create a blank mask
    mask = np.zeros((scene.shape[0], scene.shape[1]), dtype=np.uint8)

    # Iterate over all segmentations in the frame and add them to the mask
    for prediction in frame_predictions['predictions']:
        if 'points' in prediction:
            # Extract the x and y coordinates from the points
            points = np.array([[point['x'], point['y']] for point in prediction['points']], dtype=np.int32)
            # Fill the polygon defined by the points with white color (255)
            cv2.fillPoly(mask, [points], color=255)

    # Create a color version of the mask
    colored_mask = cv2.merge([mask, mask, mask])

    # Apply the mask to the scene with some transparency
    alpha = 0.3  # Transparency factor
    annotated_image = cv2.addWeighted(scene, 1, colored_mask, alpha, 0)

    return annotated_image



# Manually process the video
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = callback(frame, index)
    out.write(annotated_frame)
    index += 1

    print(f"Processed frame {index}")

cap.release()
out.release()

print("Video processing completed.")
