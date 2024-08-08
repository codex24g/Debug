import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
        
        # Load class names
        with open('class_names.json', 'r') as f:
            self.class_names = json.load(f)
        self.class_names_list = list(self.class_names.keys())

    def transform(self, frame):
        # Convert the frame to a format suitable for OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Increase the bounding box size by 20% to ensure full face coverage
            x = max(x - int(0.2 * w), 0)
            y = max(y - int(0.2 * h), 0)
            w = int(w * 1.4)
            h = int(h * 1.4)
            
            # Extract the face region
            face_img = img[y:y+h, x:x+w]
            pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((224, 224))
            processed_image = np.expand_dims(np.array(pil_image) / 255.0, axis=0)
            
            # Predict the class for the face
            predictions = self.model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = self.class_names_list[predicted_class_index]
            probability = np.max(predictions)
            
            # Get attributes for the predicted class
            attributes = self.class_names.get(predicted_class, {})
            drink_preference = attributes.get('drink_preference', 'N/A')
            dietary_restrictions = attributes.get('dietary_restrictions', 'N/A')
            
            # Draw the bounding box and label
            color = (0, 255, 0)  # Green
            label = f"{predicted_class} | {probability:.2f} | Drink: {drink_preference} | Diet: {dietary_restrictions}"
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        return img

def run():
    st.title("Real-Time Staff Classification")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
