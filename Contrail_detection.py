
# debug code
import sys
sys.path.append('/Users/lucbusquin/Projects/RMS-Contrail')
# end of debug code

from roboflow import Roboflow
import supervision as sv
import cv2
import os
import sys
import time
from shapely.geometry import Polygon
import numpy as np
import folium

RMS_Contrail_path = os.path.abspath('/Users/lucbusquin/Projects/RMS-Contrail')
sys.path.insert(0, RMS_Contrail_path)

# Now you can import modules from the other repo
from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
from RMS.Formats.Platepar import Platepar
from Utils.FOVKML import fovKML

# Print the current working directory
print("Current working directory:", os.getcwd())
pp_path = '/Users/lucbusquin/Projects/RMS_data/InferenceTest/platepar_cmn2010.cal'
pp = Platepar()
pp.read(pp_path)

# Verify video file existence
#video_path = 'C:/Users/jerom/OneDrive/PS232/Persistent_contrail - 1718392244170.mp4'
video_path = '/Users/lucbusquin/Projects/RMS_data/InferenceTest/20240213-clip1.mp4'
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

# Define the ROI (example coordinates)
roi_x, roi_y, roi_w, roi_h = 200, 200, 1720, 880

# Prepare to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output_Persistent_contrails.mp4', fourcc, fps, (width, height))
out = cv2.VideoWriter('/Users/lucbusquin/Projects/RMS_data/InferenceTest/output_Persistent_contrails.mp4', fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Cannot create video file. Check codec and file path.")
    cap.release()
    sys.exit()

frame_count = 0
start_time = time.time()

def mask_to_polygon(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [Polygon(cnt[:, 0, :]) for cnt in contours if cnt.size >= 6]
    return polygons

# Create a folium map
map_center = [34, -112]
mymap = folium.Map(location=map_center, zoom_start=10)

# Adjusting the NMS and confidence thresholds
nms_threshold = 0.3
confidence_threshold = 0.03

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count == 1000:
        print("No more frames to read or error reading a frame.")
        break

    print(f"Processing frame {frame_count}...")
    
    # Crop frame to ROI
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    cv2.imwrite('temp_frame.jpg', roi_frame)

    # Perform prediction using Roboflow
    try:
        result = model.predict('temp_frame.jpg', confidence=confidence_threshold).json()
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    # Convert Roboflow response to supervision format
    detections = sv.Detections.from_inference(result)

    # Apply Non-Maximum Suppression with the adjusted NMS threshold
    #detections = sv.apply_nms(detections, nms_threshold)

    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    # Annotate frame with labels and masks
    #annotated_roi_frame = mask_annotator.annotate(scene=roi_frame, detections=detections)

    # Check if there are any detections
    if detections.xyxy.size > 0:
        # Extract masks from detections and convert to polygons
        for xyxy, mask, confidence, class_id, data in zip(
            detections.xyxy, detections.mask, detections.confidence, detections.class_id, detections.data):
            if mask is not None:
                print(confidence)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = [Polygon(cnt[:, 0, :]) for cnt in contours if cnt.size >= 6]

                cv2.drawContours(roi_frame, contours, -1, (0, 255, 0), 2)  # Green contours for masks
                # Convert polygons to lat, lon and plot
                for polygon in polygons:
                    geo_polygon = []
                    for coord in np.array(polygon.exterior.coords):
                        lat, lon = XyHt2Geo(pp, coord[0] + roi_x, coord[1] + roi_y, 11000)
                        geo_polygon.append((lat, lon))
                    # Add polygon to map
                    folium.Polygon(locations=geo_polygon, color='blue').add_to(mymap)

    # Place annotated ROI frame back into the original frame
    frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_frame

    # Write the annotated frame to output video
    out.write(frame)
    print(f"Frame {frame_count} written to video.")

    frame_count += 1

end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time} seconds.")

# Save the map to an HTML file
mymap.save('/Users/lucbusquin/Projects/RMS_data/InferenceTest/polygons_map.html')

# Cleanup
cap.release()
out.release()
print("Video processing complete, resources released.")