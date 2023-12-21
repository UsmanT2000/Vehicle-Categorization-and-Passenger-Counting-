from ultralytics import YOLO
from ultralytics.solutions import object_counter
from collections import defaultdict
import time
import streamlit as st
import cv2
import numpy as np
import csv

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


# def display_tracker_options():
#     display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
#     is_display_tracker = True if display_tracker == 'Yes' else False
#     if is_display_tracker:
#         tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
#         return is_display_tracker, tracker_type
#     return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image):
    #, tracker=None
    # is_display_tracking=None,
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    #if is_display_tracking:
    res = model.track(image, conf=conf, persist=True)
    #else:
        # Predict the objects in the image using the YOLOv8 model
        #res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """

    #is_display_tracker, tracker = display_tracker_options()


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    #is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                    #is_display_tracker,tracker
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))

with open('object_counts.csv', 'w', newline='') as csvfile:
    fieldnames = ['frame_number', 'in_count', 'out_count']  # Update fieldnames
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    #is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            
            track_history = defaultdict(lambda: [])
            assert vid_cap.isOpened(), "Error reading video file"
            counter = object_counter.ObjectCounter()  # Init Object Counter
            region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
            counter.set_args(view_img=True,
                            reg_pts=region_points,
                            classes_names=model.names,
                            draw_tracks=True)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, im0 = vid_cap.read()
                if success:
                    with open('object_counts.csv', 'a', newline='') as csvfile:  # Open CSV within the loop
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        in_count = counter.in_counts
                        out_count = counter.out_counts
                        frame_number = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
                        writer.writerow({'frame_number': frame_number, 'in_count': in_count, 'out_count': out_count})
                    image = cv2.resize(im0, (720, int(720*(9/16))))
                    results = model.track(im0, persist=True)
                    im0 = counter.start_counting(im0, results)
                    if results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                            # Get the boxes and track IDs
                            boxes = results[0].boxes.xywh.cpu()
                            track_ids = results[0].boxes.id.int().cpu().tolist()

                            # Visualize the results on the frame
                            annotated_frame = results[0].plot()

                            # Plot the tracks
                            for box, track_id in zip(boxes, track_ids):
                                x, y, w, h = box
                                track = track_history[track_id]
                                track.append((float(x), float(y)))  # x, y center point
                                if len(track) > 30:  # retain 90 tracks for 90 frames
                                    track.pop(0)

                                # Draw the tracking lines
                                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        
                                st_frame.image(annotated_frame,
                                            caption='Detected Video',
                                            channels="BGR",
                                            use_column_width=True
                                            )

                    #is_display_tracker,tracker
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
