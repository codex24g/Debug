import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json

def load_model_and_class_names():
    try:
        model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
    
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        class_names_list = list(class_names.keys())
        st.write("Class names loaded successfully.")
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        class_names = {}
        class_names_list = []
    
    return model, class_names, class_names_list

def process_frame(frame, model, class_names, class_names_list):
    if model is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        x = max(x - int(0.2 * w), 0)
        y = max(y - int(0.2 * h), 0)
        w = int(w * 1.4)
        h = int(h * 1.4)
        
        face_img = frame[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((224, 224))
        processed_image = np.expand_dims(np.array(pil_image) / 255.0, axis=0)
        
        try:
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names_list[predicted_class_index]
            probability = np.max(predictions)
            
            attributes = class_names.get(predicted_class, {})
            drink_preference = attributes.get('drink_preference', 'N/A')
            dietary_restrictions = attributes.get('dietary_restrictions', 'N/A')
            
            color = (0, 255, 0)
            label = f"{predicted_class} | {probability:.2f} | Drink: {drink_preference} | Diet: {dietary_restrictions}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    return frame

def main():
    st.title("Real-Time Staff Classification with OpenCV")

    # Load model and class names
    model, class_names, class_names_list = load_model_and_class_names()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the default camera (0)
    
    # Streamlit video display
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Process the frame
        processed_frame = process_frame(frame, model, class_names, class_names_list)
        
        # Convert the frame to RGB for Streamlit
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        
        # Allow user to stop the video stream
        if st.button("Stop Stream"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
