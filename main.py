import streamlit as st
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image
import numpy as np

from util import classify, set_background


# ⛑️ Patch: Custom class to ignore unsupported 'groups' parameter
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if it exists
        super().__init__(*args, **kwargs)

# Load with patch
custom_objects = {'DepthwiseConv2D': PatchedDepthwiseConv2D}


# 🖼️ Set background
set_background('./bgs/bg4.jpg')

# 🏷️ Title & Header
# st.title('Pneumonia classification')
# st.header('Please upload a chest X-ray image')

st.markdown("""
    <h1 style='text-align: center; color: #FFFFFF; font-size: 100px;'>
        Pneumonia Classification
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='text-align: center; color: #FFFFFF; font-size: 40px;'>
        Please Upload a Chest X-Ray Image
    </h2>
""", unsafe_allow_html=True)





# 📤 Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# 🧠 Load model (patched loader)
model = load_model(r"E:\Pneumonia classifier\model\pneumonia_classifier.h5", custom_objects=custom_objects)

# 📄 Load class labels
with open('./model/labels.txt', 'r') as f:
    class_names = [a.strip().split(' ')[1] for a in f.readlines()]

# 🖼️ Display + predict
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
