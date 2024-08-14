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

def main():
    # Pre-defined paths
    rms_contrail_path = '/Users/jerom/Projects/RMS-Contrail'
    rms_data_path = '/Users/jerom/Projects/RMS_data/InferenceTest'
    video_path = 'C:/Users/jerom/Projects/Input_video/shorty_cont.mp4'

    # Set up paths
    sys.path.append(rms_contrail_path)

    # Import RMS modules
    try:
        from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
        from RMS.Formats.Platepar import Platepar
    except ImportError as e:
        print(f"Error importing RMS modules: {e}")
        sys.exit()

    # Load platepar file
    pp_path = os.path.join(rms_data_path, 'platepar_cmn2010.cal')
    pp = Platepar()
    try:
        pp.read(pp_path)
    except FileNotFoundError:
        print(f"Error: The file {pp_path} was not found.")
        sys.exit()

    # Verify video file existence
    if not os.path.exists(video_path):
        print(f"Error: The video file {video_path} does not exist.")
        sys.exit()

    # Initialize Roboflow with your API key
    api_key = "yCdhJRWuNLpR7kARUOTD"  # API key
    rf = Roboflow(api_key=api_key)
    
    # Handle the workspace and project manually if necessary
    try:
        workspace = rf.workspace()
        project = workspace.project("contrails-0lwhn")
        model = project.version(2).model
    except KeyError:
        print("Error: Workspace or project could not be accessed. Please check your API key and workspace permissions.")
        sys.exit()

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit()

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the ROI
    roi_x, roi_y, roi_w, roi_h = 200, 200, 1720, 880

    # Prepare to save the output video
    output_video_path = os.path.join(rms_data_path, 'output_Persistent_contrails.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Cannot create video file. Check codec and file path.")
        cap.release()
        sys.exit()

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

    # Define the camera location
    camera_lat = 33.580453
    camera_lon = -112.019888

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

        # Create a new map for this frame with an adjustable zoom level
        zoom_level = 10  # Adjust this value to zoom in or out
        mymap = folium.Map(location=[camera_lat, camera_lon], zoom_start=zoom_level)

        # Add distance rings from 5 km to 40 km and label them
        for radius in range(5, 45, 5):  # Radius from 5 km to 40 km
            # Add the circle
            folium.Circle(
                location=[camera_lat, camera_lon],
                radius=radius * 1000,  # Convert km to meters
                color='red',
                fill=False
            ).add_to(mymap)
            
            # Add label to the ring
            label_location = [
                camera_lat + (radius / 111),  # Rough approximation to place the label outside the circle
                camera_lon
            ]
            folium.map.Marker(
                label_location,
                icon=folium.DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 12pt; color : red">{radius} km</div>',
                )
            ).add_to(mymap)

        # Check if there are any detections
        if detections.xyxy.size > 0:
            for xyxy, mask, confidence, class_id, data in zip(
                detections.xyxy, detections.mask, detections.confidence, detections.class_id, detections.data):
                
                # Make sure the mask is not None and has contours
                if mask is not None:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:  # Check if contours were found
                        polygons = [Polygon(cnt[:, 0, :]) for cnt in contours if cnt.size >= 6]
                        
                        for polygon in polygons:  # Ensure 'polygon' is properly defined
                            cv2.drawContours(roi_frame, contours, -1, (0, 255, 0), 2)  # Green contours for masks
                            
                            # Convert polygons to lat, lon and plot for multiple altitudes
                            altitudes = [9000, 10000, 11000, 12000]  # List of altitudes in meters
                            colors = ['blue', 'green', 'orange', 'purple']  # Corresponding colors for each altitude
                            for altitude, color in zip(altitudes, colors):
                                geo_polygon = []
                                for coord in np.array(polygon.exterior.coords):
                                    lat, lon = XyHt2Geo(pp, coord[0] + roi_x, coord[1] + roi_y, altitude)
                                    geo_polygon.append((lat, lon))
                                folium.Polygon(locations=geo_polygon, color=color, fill=True, fill_color=color, fill_opacity=0.2, tooltip=f'Altitude: {altitude}m').add_to(mymap)

        # Place annotated ROI frame back into the original frame
        frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_frame

        # Write the annotated frame to output video
        out.write(frame)
        print(f"Frame {frame_count} written to video.")

        # Save the map as an image
        map_filename = os.path.join(rms_data_path, f'map_frame_{frame_count}')
        save_map_as_image(mymap, map_filename)
        map_images.append(map_filename + '.png')

        frame_count += 1

    end_time = time.time()
    print(f"Processed {frame_count} frames in {end_time - start_time} seconds.")

    # Create a video from the map images
    map_video_path = os.path.join(rms_data_path, 'map_timelapse.mp4')
    map_video = cv2.VideoWriter(map_video_path, fourcc, fps, (800, 600))

    for image in map_images:
        img = cv2.imread(image)
        map_video.write(img)

    map_video.release()

    # Cleanup
    cap.release()
    out.release()
    print("Video processing complete, resources released.")

if __name__ == "__main__":
    main()
