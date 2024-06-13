import random
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

@st.cache(allow_output_mutation = True)

def load_model():
    model = tf.keras.models.load_model("cnnMelanomaX.hdf5")
    return model
with st.spinner("loading model"):
    model = load_model()
    
st.write("""
         #  Melanoma Classification
         """)

file = st.file_uploader("upload a file: ", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img_reshape = image[np.newaxis,...]
    predict = model.predict(img_reshape)
    
    return predict

class_names = ['benign', "malignant"]

if file is None:
    st.text("Upload an image file!1!1!!!!!11")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    # st.write(predictions[0,0])
    # st.write(score)
    benign = predictions[0,0]
    malignant = predictions[0,1]
    
    if benign>malignant:
        st.write("Detected: Benign, confidence: "+ str(benign))
    else:
        st.write("Detected: Malignant, confidence: "+ str(malignant))
    
    # print(
    # "This image most likely belongs to {predictions} with a {:.2f} percent confidence."
    # .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )