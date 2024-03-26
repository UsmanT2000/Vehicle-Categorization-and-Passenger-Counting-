from ultralytics import YOLO
from ultralytics.solutions import object_counter
from collections import defaultdict
import time
import streamlit as st
import cv2
import numpy as np
import csv
import pandas as pd
import settings
import datetime
import matplotlib.pyplot as plt
import sqlite3
import hashlib


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
def plot_donut_chart(data1, data2, title):
    labels = ['Outflow Count', 'Inflow Count']
    sizes = [data1['Out Count'], data2['In Count']]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    st.pyplot(fig)


def plot_histogram(data1, data2, title):
    # Data
    labels = ['Out Count', 'In Count']
    sizes = [data1['Out Count'], data2['In Count']]
    
    # Define colors for each bar
    colors = ['blue', 'orange']
    
    # Create histogram
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=colors)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

def plot_vehicle_histogram(data, title):
    # Data
    labels = data.keys()
    sizes = data.values()
    
    # Define colors for each bar
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'yellow']
    
    # Create histogram
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=colors)
    ax.set_xlabel('Vehicle Types')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    csv_filename = f"object_counts_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['frame_number', 'in_count', 'out_count', 'vehicle_type']  # Update fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()   # Write the header row
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    #is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Generate Dashboard'):
        df = pd.read_csv('test2.csv')

        last_row = df.iloc[-1]  # Get the last row of the DataFrame
        out_count = last_row['out_count']
        in_count = last_row['in_count']
        vehicle_type = last_row['vehicle_type']

        out_count = int(out_count)
        in_count = int(in_count)
        out_info_counts = {'Out Count': out_count}
        in_info_counts = {'In Count': in_count}
        vehicle_counts = defaultdict(int)
        vehicle_counts[vehicle_type] += 1
        
        st.title("Outflow and Inflow Info Donut Chart")
        plot_donut_chart(out_info_counts,in_info_counts ,"Outflow and Inflow Info Distribution")

        st.title("Outflow and Inflow Histogram")
        plot_histogram(out_info_counts,in_info_counts,"Outflow and Inflow Histogram")

        plot_vehicle_histogram(vehicle_counts, "Vehicle Types Distribution")


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
                    with open(csv_filename, 'a', newline='') as csvfile:  # Open CSV within the loop
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        in_count = counter.in_counts
                        out_count = counter.out_counts
                        frame_number = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
                        results = model.track(im0, persist=True)
                        vehicle_type = model.names[3]
                        writer.writerow({'frame_number': frame_number, 'in_count': in_count, 'out_count': out_count, 'vehicle_type': vehicle_type})
                    image = cv2.resize(im0, (720, int(720*(9/16))))
                    
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

def create_connection():
    """
    Create a database connection to the SQLite database.

    Returns:
        conn: SQLite database connection object
    """
    conn = None
    try:
        conn = sqlite3.connect('database.db')  # Replace 'database.db' with your database file name
        return conn
    except sqlite3.Error as e:
        print(e)
    
    return conn

def create_user_table(conn):
    """
    Create a 'users' table in the database if it doesn't exist.

    Parameters:
        conn (sqlite3.Connection): SQLite database connection object
    """
    try:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                        )''')
        conn.commit()
    except sqlite3.Error as e:
        print(e)

def authenticate_user(conn, username, password):
    """
    Authenticate a user based on the provided username and password.

    Parameters:
        conn (sqlite3.Connection): SQLite database connection object
        username (str): Username provided by the user
        password (str): Password provided by the user

    Returns:
        bool: True if the user is authenticated, False otherwise
    """
    # Hash the password provided by the user
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Execute a SQL query to fetch the user with the provided username and hashed password
    cursor = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))

    # Fetch one row from the result set
    user = cursor.fetchone()

    # If the user exists in the database, return True (authenticated), otherwise return False
    return user is not None


def login_page(conn):
    """
    Display the login page.

    Parameters:
        conn (sqlite3.Connection): SQLite database connection object
    """
    st.title("Login Page")

    # Username and password input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if not username or not password:
            st.error("Username and password are required")
        else:
            # Authenticate the user
            authenticated = authenticate_user(conn, username, password)
            if authenticated:
                st.session_state.is_logged_in = True
            else:
                st.error("Invalid username or password")