# Python In-built packages
import hashlib
from pathlib import Path

# External packages
import streamlit as st
import sqlite3
# Local Modules
import helper
import settings
# Setting page layout
st.set_page_config(
    page_title="Lucid Transit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to register a new user
def register_user(conn, username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()

# Registration page content
def registration_page(conn):
    st.title("Registration Compartment")

    # Username and password input fields
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")

    # Registration button
    if st.button("Register"):
        if not username or not password:
            st.error("Username and password are required")
        else:
            # Check if the username already exists
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            if cursor.fetchone() is not None:
                st.error("Username already exists. Please choose a different one.")
            else:
                # Register the user
                register_user(conn, username, password)
                st.success("Registration successful. You can now log in.")

# Main page content
def main_page():
    # Main page heading
    st.title("Vehicle Categorisation using YOLOv8")

    # Sidebar
    st.sidebar.header("ML Model Config")

    # Model Options
    model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])

    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

    # Selecting Detection Or Segmentation
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Video Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    if model_type == 'Passenger Counting' and source_radio == settings.VIDEO:
        helper.play_passengercount_video(confidence, model)
    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)
    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)
    else:
        st.error("Please select a valid source type!")

# Main function
def main():
    conn = helper.create_connection()  # Create a database connection
    helper.create_user_table(conn)  # Create the user table if it doesn't exist

    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    
    if not st.session_state.is_logged_in:
        helper.login_page(conn)
        registration_page(conn)  # Display the registration page
    else:
        main_page()

if __name__ == "__main__":
    main()
