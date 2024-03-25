# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Lucid Transit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Login page content
def login_page():
    st.title("Login Page")

    # Username and password input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if username == "lucidtransit" and password == "password":
            st.session_state.is_logged_in = True  # Set login state to True
        else:
            st.error("Invalid username or password")

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

    if source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)
    
    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)

    else:
        st.error("Please select a valid source type!")

def main():
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    
    if not st.session_state.is_logged_in:
        login_page()
    else:
        main_page()

if __name__ == "__main__":
    main()
