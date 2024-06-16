from roboflow import Roboflow
import supervision as sv
import cv2
import os
import sys
import time

# Print the current working directory
print("Current working directory:", os.getcwd())

# Verify video file existence
video_path = 'C:/Users/jerom/OneDrive/PS232/Persistent_contrail - 1718392244170.mp4'
if not os.path.exists(video_path):
    print(f"Error: The video file {video_path} does not exist.")
    sys.exit()

rf = Roboflow(api_key="yCdhJRWuNLpR7kARUOTD")
project = rf.workspace().project("contrails-0lwhn")
model = project.version(2).model

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    sys.exit()
else:
    print("Video file opened successfully.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_Persistent_contrails.mp4', fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Cannot create video file. Check codec and file path.")
    cap.release()
    sys.exit()

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read or error reading a frame.")
        break

    print(f"Processing frame {frame_count}...")
    cv2.imwrite('temp_frame.jpg', frame)

    # Perform prediction using Roboflow
    try:
        result = model.predict('temp_frame.jpg', confidence=70).json()
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    # Convert Roboflow response to supervision format
    detections = sv.Detections.from_inference(result)

    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    # Annotate frame with labels and masks
    annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=[item["class"] for item in result["predictions"]])

    # Write the annotated frame to output video
    out.write(annotated_image)
    print(f"Frame {frame_count} written to video.")

    frame_count += 1

end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time} seconds.")

# Cleanup
cap.release()
out.release()
print("Video processing complete, resources released.")
