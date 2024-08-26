import streamlit as st
from streamlit.logger import get_logger
import subprocess
from PIL import Image

def run_script(script_name):
    subprocess.run(["streamlit", "run", script_name])

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Python Code Similarity Detection and Clustering Tool",
        page_icon="logo/logo.png",  # Set your logo image as the page icon
    )
    
    # Check if session state for images is initialized
    if 'cover_photo' not in st.session_state:
        st.session_state.cover_photo = Image.open("logo/lbg.png")  # Update with the path to your cover photo image

    if 'cspc_logo' not in st.session_state:
        st.session_state.cspc_logo = Image.open("logo/cspc.png")

    if 'ccs_logo' not in st.session_state:
        st.session_state.ccs_logo = Image.open("logo/ccs.png")

    if 'l2_logo' not in st.session_state:
        st.session_state.l2_logo = Image.open("logo/l2.png")
    
    # Display the cover photo image
    st.image(st.session_state.cover_photo, use_column_width=True)
    
    # Display the title
    st.markdown("<h2 style='text-align: center; margin: 20px;'>Python Code Similarity Detection and Clustering Tool</h2>", unsafe_allow_html=True)
    
    st.write("---")
    st.markdown(
        """<div style='text-align: justify; text-indent: 30px;'>
        Welcome to Python Code Similarity Detection and Clustering Tool by ORION. This tool is developed using a hybrid method that combines AST-based and text-based approaches with 
        the K-Means algorithm to detect code similarity in Python.
        </div><br>""",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div style='text-align: justify; text-indent: 30px;'>
        Detecting similarity between code submissions can be time-consuming and difficult. To address this problem, this tool offers an innovative method for detecting code similarity,
        focusing on both the structural representation and code text information of Python code provided by Abstract Syntax Trees (ASTs) and a text-based approach. By leveraging these
        attributes, the method seeks to achieve a comprehensive detection of similarity in Python codes, capturing both structural patterns and textual features. Additionally, the computational
        efficiency of the K-Means algorithm will be utilized in clustering Python codes that exhibit similarity in their code text and structure, streamlining the process of identifying
        similarities and differences with in a set of code submissions.
        </div>""",
        unsafe_allow_html=True
    )
    st.write("---")
    
    # Display the images as a footer
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(st.session_state.cspc_logo, use_column_width=True)
    with col2:
        st.image(st.session_state.ccs_logo, use_column_width=True)
    with col3:
        st.image(st.session_state.l2_logo, use_column_width=True)
    
    # Add some spacing and align images properly using CSS
    st.markdown("""
    <style>
    .stImage {
        display: flex;
        justify-content: center;
        margin-bottom: 0 !important;
    }
    .stColumn > div {
        padding: 0px 5px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()
