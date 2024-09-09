
import sys
import os
import cv2
import numpy as np
from roboflow import Roboflow
import folium
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import io
import time
from selenium import webdriver
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# Kalman Filter class
class KalmanFilter:
    def __init__(self, point):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.statePre = np.array([point[0], point[1], 0, 0], np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, point):
        return self.kalman.correct(np.array([[np.float32(point[0])], [np.float32(point[1])]]))

# Tracking and ID assignment logic
class ObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 1
        self.frame_threshold = 30  # maximum distance between detections in consecutive frames

    def track_objects(self, points):
        updated_trackers = {}

        if len(self.trackers) == 0:
            # Initialize Kalman filters for all objects in the first frame
            for point in points:
                self.trackers[self.next_id] = KalmanFilter(point)
                updated_trackers[self.next_id] = point
                self.next_id += 1
        else:
            # Predict the positions of existing trackers
            predicted_points = []
            tracker_ids = []
            for tracker_id, tracker in self.trackers.items():
                predicted_point = tracker.predict()
                predicted_points.append(predicted_point[:2])
                tracker_ids.append(tracker_id)

            # Use Hungarian algorithm to associate current frame points with predicted points
            cost_matrix = np.linalg.norm(np.array(predicted_points)[:, np.newaxis] - np.array(points), axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Update existing trackers with corresponding detections
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.frame_threshold:
                    tracker_id = tracker_ids[r]
                    self.trackers[tracker_id].correct(points[c])
                    updated_trackers[tracker_id] = points[c]

            # Create new trackers for unmatched detections
            unmatched_points = set(range(len(points))) - set(col_ind)
            for idx in unmatched_points:
                self.trackers[self.next_id] = KalmanFilter(points[idx])
                updated_trackers[self.next_id] = points[idx]
                self.next_id += 1

        self.trackers = {tracker_id: self.trackers[tracker_id] for tracker_id in updated_trackers}
        return updated_trackers

# Save map as image
def save_map_as_image(polygons, object_ids, filename):
    map_center = [33.58, -112.019]
    mymap = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron')

    # Create a mapping from IDs to letters (Contrail A, Contrail B, etc.)
    id_to_label = {obj_id: chr(65 + idx) for idx, obj_id in enumerate(object_ids)}  # 65 is ASCII for 'A'

    for polygon, obj_id in zip(polygons, object_ids):
        label = f"Contrail {id_to_label[obj_id]}"  # e.g., Contrail A, Contrail B
        folium.Polygon(locations=polygon, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(mymap)
        folium.Marker(
            location=polygon[0],  # Place label at the first point of the polygon
            popup=label,  # Set the popup text to Contrail A, B, etc.
            icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black">{label}</div>')  # Display the label as text
        ).add_to(mymap)

    map_html_path = filename + '.html'
    mymap.save(map_html_path)

    # Use selenium to take a screenshot of the HTML
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(800, 600)
    driver.get('file://' + os.path.abspath(map_html_path))

    png = driver.get_screenshot_as_png()
    driver.quit()

    # Save the PNG image
    img = Image.open(io.BytesIO(png))
    img.save(filename + '.png')


def process_frame(frame_index, frame_predictions, pp_path, tracker):
    sys.path.append('C:/Users/jerom/Projects/RMS-Contrail')
    from RMS.Astrometry.ApplyAstrometry import XyHt2Geo
    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.read(pp_path)

    polygons = []
    current_points = []

    for prediction in frame_predictions['predictions']:
        if 'points' in prediction:
            points = np.array([[point['x'], point['y']] for point in prediction['points']], dtype=np.int32)
            geo_polygon = []
            for point in points:
                lat, lon = XyHt2Geo(pp, point[0], point[1], 11000)
                geo_polygon.append((lat, lon))
            if geo_polygon:
                polygons.append(geo_polygon)
                current_points.append((geo_polygon[0][0], geo_polygon[0][1]))  # Use the first point of the polygon for tracking

    # Track objects and assign IDs
    tracked_objects = tracker.track_objects(current_points)

    map_filename = f'/Users/jerom/Projects/RMS_data/InferenceTest/map_frame_{frame_index}'
    save_map_as_image(polygons, list(tracked_objects.keys()), map_filename)
    return map_filename + '.png'

def main():
    print("Current working directory:", os.getcwd())
    rms_path = 'C:/Users/jerom/Projects/RMS-Contrail'
    sys.path.append(rms_path)

    from RMS.ConfigReader import parse
    config_file = os.path.join(rms_path, '.config')
    config = parse(config_file)
    pp_path = '/Users/jerom/Projects/RMS_data/InferenceTest/platepar_cmn2010.cal'

    rf = Roboflow(api_key="6Kzy0L8WP8idrygkLOUr")
    workspace = rf.workspace("contrailcast")
    project = workspace.project("contrails-0lwhn-ibga0")
    model = project.version(1).model
    video_path = 'C:/Users/jerom/Projects/Input_video/dual_cont_det.mp4'

    job_id, signed_url, expire_time = model.predict_video(video_path, fps=30, prediction_type="batch-video")
    results = model.poll_until_video_results(job_id)

    tracker = ObjectTracker()
    map_images = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_index, frame_predictions, pp_path, tracker)
                   for frame_index, frame_predictions in enumerate(results['contrails-0lwhn-ibga0'])]
        for future in futures:
            map_images.append(future.result())

    if map_images:
        first_image_path = map_images[0]
        first_img = cv2.imread(first_image_path)
        if first_img is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            map_video = cv2.VideoWriter('/Users/jerom/Projects/RMS_data/InferenceTest/map_timelapse_gts.mp4', 
                                        fourcc, 30, (first_img.shape[1], first_img.shape[0]))
            for image in map_images:
                img = cv2.imread(image)
                if img is None:
                    continue
                map_video.write(img)
            map_video.release()

if __name__ == "__main__":
    main()
