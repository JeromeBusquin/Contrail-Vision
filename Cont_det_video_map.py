import sys
import os
import time
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from shapely.geometry import Polygon
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io

# Add necessary paths
sys.path.append('/Users/jerom/Projects/RMS-Contrail')
RMS_Contrail_path = os.path.abspath('/Users/jerom/Projects/RMS-Contrail')
sys.path.insert(0, RMS_Contrail_path)

# Import RMS modules
from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
from RMS.Formats.Platepar import Platepar

# Print the current working directory
print("Current working directory:", os.getcwd())
pp_path = '/Users/jerom/Projects/RMS_data/InferenceTest/platepar_cmn2010.cal'
pp = Platepar()
pp.read(pp_path)

# Verify video file existence
video_path = 'C:/Users/jerom/Projects/Input_video/shorty_cont.mp4'
if not os.path.exists(video_path):
    print(f"Error: The video file {video_path} does not exist.")
    sys.exit()

# Initialize Roboflow
rf = Roboflow(api_key="yCdhJRWuNLpR7kARUOTD")
project = rf.workspace().project("contrails-0lwhn")
model = project.version(2).model

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    sys.exit()
else:
    print("Video file opened successfully.")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the ROI
roi_x, roi_y, roi_w, roi_h = 200, 200, 1720, 880

# Prepare to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/Users/jerom/Projects/RMS_data/InferenceTest/output_Persistent_contrails.mp4', fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Cannot create video file. Check codec and file path.")
    cap.release()
    sys.exit()

# Function to save map as image
def save_map_as_image(map_obj, filename):
    # Save map as HTML
    map_obj.save(filename + '.html')
    
    # Use Selenium to capture the map as an image
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get('file://' + os.path.abspath(filename + '.html'))
    time.sleep(2)  # Wait for map to load
    
    # Capture the image
    png = driver.get_screenshot_as_png()
    driver.quit()
    
    # Save the image
    im = Image.open(io.BytesIO(png))
    im.save(filename + '.png')

frame_count = 0
start_time = time.time()

# Adjusting the confidence threshold
confidence_threshold = 5

# List to store map image filenames
map_images = []

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

    # Create a new map for this frame
    map_center = [34, -112]
    mymap = folium.Map(location=map_center, zoom_start=10)

    # Check if there are any detections
    if detections.xyxy.size > 0:
        for xyxy, mask, confidence, class_id, data in zip(
            detections.xyxy, detections.mask, detections.confidence, detections.class_id, detections.data):
            if mask is not None:
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
                    folium.Polygon(locations=geo_polygon, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(mymap)

    # Place annotated ROI frame back into the original frame
    frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_frame

    # Write the annotated frame to output video
    out.write(frame)
    print(f"Frame {frame_count} written to video.")

    # Save the map as an image
    map_filename = f'/Users/jerom/Projects/RMS_data/InferenceTest/map_frame_{frame_count}'
    save_map_as_image(mymap, map_filename)
    map_images.append(map_filename + '.png')

    frame_count += 1

end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time} seconds.")

# Create a video from the map images
map_video = cv2.VideoWriter('/Users/jerom/Projects/RMS_data/InferenceTest/map_timelapse.mp4', fourcc, fps, (800, 600))

for image in map_images:
    img = cv2.imread(image)
    map_video.write(img)

map_video.release()

# Cleanup
cap.release()
out.release()
print("Video processing complete, resources released.")