import keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow.compat.v2 as tf
tf.disable_v2_behavior()

st.title("Breast Cancer Prediction")
st.header("Upload a image for classification as Cancer or No-Cancer")
st.write("\n")

uploaded_file = st.file_uploader("Choose a histopathology image ...", type="tif")
safe_html="""
<h2 style="color:green ;text-align:center;">The tissue is Healthy</h2>
</div>
"""
danger_html="""
<h2 style="color:red ;text-align:center;">The tissue is Cancerous</h2>
</div>
"""
def predict_cls(img, weights_file):
    model= keras.models.load_model(weights_file)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #return np.argmax(prediction) # return position of the highest probability
    return prediction


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', width=None)
    st.write("")
    st.write("Classifying...")
    prediction = predict_cls(image, 'xception30.h5')
    pred='{0:{1}f}'.format(prediction[0][0],4)
    
    if prediction[0][0]>=0.05:
        st.markdown(danger_html,unsafe_allow_html=True)
        st.write("\n")
        st.write("The Probability of having Cancer is: {}".format(pred))
    else:
        st.markdown(safe_html,unsafe_allow_html=True)
        st.write("\n")
        st.write("The Probability of having Cancer is: {}".format(pred))
    
