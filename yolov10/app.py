import os
import base64
import streamlit as st
from PIL import Image
import requests
import io

from utils import save_upload_file, delete_file
from models.yolov10.detector import inference, inference_sahi
from config.model_config import Detector_Config
from components.streamlit_footer import footer
import helper

@st.cache_data(max_entries=1000)
def process_inference(image_path):
    # Inference normally
    result_img = inference(
        image_path,
        weight_path=Detector_Config().weight_path
    )
    # Inference SAHI
    result_img_sahi = inference_sahi(
        Image.open(image_path),
        weight_path=Detector_Config().weight_path
    )
    result_img_sahi.export_visuals(export_dir="demo_data/")
    return result_img, result_img_sahi

def main():
    st.set_page_config(
        page_title="YOLOv10 Detection Demo",
        page_icon='static/artaxor.png',
        layout="wide"
    )

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title(':sparkles: :blue[YOLOv10] Detection Demo')
        st.text('Model: Pre-trained YOLOv10n')

    with col2:
        logo_img = open("static/artaxor.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <img src="data:image/png;base64,{logo_base64}" width="full">
            """,
            unsafe_allow_html=True,
        )
        
    # Model Options
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
        
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", ["Image", "YouTube"])
    
    st.write("Hemiptera (côn trùng nửa cánh) bao gồm nhiều loài gây hại nghiêm trọng cho cây trồng, như rệp, bọ xít, và bọ phấn.")
    st.write("Coleoptera (côn trùng cánh cứng) bao gồm các loài gây hại nghiêm trọng như bọ hung, sâu đục thân, và bọ nhảy.")
    
    if source_radio == "Image":
        uploaded_img = st.file_uploader('__Input your image__', type=['jpg', 'jpeg', 'png'])

        st.divider()

        # if uploaded_img:
        if st.button("Inference"):
                result_img, _ = process_inference(uploaded_img)
        else:
            result_img = None
            
        col3, col4 = st.columns(2)
        st.markdown('**Detection result**')
        if result_img is not None:
            with col3:
                st.image(result_img, width=600, caption="Inference normal")
            with col4:
                st.image("demo_data/prediction_visual.png", width=600, caption="Inference SAHI")

        footer()
    
    elif source_radio == 'YouTube':
        helper.play_youtube_video(confidence, Detector_Config().weight_path)
        footer()
    else:
        st.error("Please select a valid source type!")

if __name__ == '__main__':
    # if not os.path.exists(Detector_Config.weight_path):
    #     download_model()
    main()