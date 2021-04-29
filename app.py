from image_classification import predict_cls
from PIL import Image, ImageOps
import streamlit as st

st.title("Breast Cancer Prediction")
st.header("Upload a image for classification as Cancer or No-Cancer")
st.write("\n")

uploaded_file = st.file_uploader("Choose a histopathology image ...", type="tif")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', width=None)
    st.write("")
    st.write("Classifying...")
    prediction = predict_cls(image, 'xception30.h5')
    
    if prediction[0][0]>=0.05:
        st.write("The tissue is Cancerous")
        st.write("\n")
        st.write("The Probability of having Cancer is: {}".format(prediction[0][0]))
    else:
        st.write("The tissue is Healthy")
        st.write("\n")
        st.write("The Probability of having Cancer is: {}".format(prediction[0][0]))
    