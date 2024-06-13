import os
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

# Load the bone age model
bone_age_model = tf.keras.models.load_model(r"D:\medical\bones age\bone_age_weights.best.h5")

# Function to load and resize image
def load_and_resize_image(image_path, target_shape=(384, 384)):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image
    resized_image = cv2.resize(image, target_shape)
    # Expand dimensions to match the desired shape (None, 384, 384, 3)
    resized_image = resized_image[np.newaxis, ...]
    return resized_image

# Function to predict bone age
def predict_bone_age(image):
    resized_image = load_and_resize_image(image)
    pred = bone_age_model.predict(resized_image)
    return pred

# Streamlit app
def main():
    st.title("Bone Age Prediction App")
    st.write("Upload one or more images of hand X-rays to predict the bone age.")

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            with NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                image_path = tmp_file.name

            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Predict bone age
            prediction = predict_bone_age(image_path)
            if prediction[0][0] <= 20:
                prediction = round(prediction[0][0], 2)
                ym = " months"
            else:
                prediction = round(prediction[0][0] / 12, 2)
                ym = " years"
            # Display the predicted bone age below the image
            ################################################################################
            st.markdown("""
            <style>
            .big-font {
                font-size:25px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            message = "Expected age for " + uploaded_file.name + " is : " + str(prediction) + ym
            styled_message = f'<p class="big-font">{message}</p>'
            st.markdown(styled_message, unsafe_allow_html=True)
            ################################################################################
            # st.write("Predicted bone age for ",uploaded_file.name," :", prediction, ym)

            # Delete the temporary file
            os.unlink(image_path)

if __name__ == "__main__":
    main()
