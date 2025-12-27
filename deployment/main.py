import numpy as np
import streamlit as st
import cv2
import json
import requests

with open('labels.json') as f:
    labels = json.load(f)
    
st.title("Cassava Disease Classification")
st.write("Drop an image of a cassava leaf to get prediction.")

uploaded_file = st.file_uploader("Choose and image....",type=['jpg','jpeg','png'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_flat = img_resized.flatten().tolist()
    payload = {
        "inputs": [
            {
                "name": "INPUT_IMAGE",
                "shape": [224, 224, 3],
                "datatype": "UINT8",
                "data": img_flat
            }
        ]
    }
    try:
        response = requests.post("http://localhost:8000/v2/models/ensemble/infer",
                                 json=payload)
        pred_probs = response.json()["outputs"][0]["data"]
        pred_idx = np.argmax(pred_probs)
        st.success(f"Prediction: {labels[str(pred_idx)]}")
    except Exception as e:
        st.error(f"Error connecting to Triton server: {e}")    