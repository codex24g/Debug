# image_classification.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths and parameters
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
img_height, img_width = 224, 224
batch_size = 8

def update_class_names(class_name, drink_preference, dietary_restrictions):
    try:
        # Load existing class names
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
        else:
            class_names = {}

        # Add or update the new class information
        class_names[class_name] = {
            "drink_preference": drink_preference,
            "dietary_restrictions": dietary_restrictions
        }

        # Sort class names alphabetically and update the dictionary
        sorted_class_names = dict(sorted(class_names.items()))

        # Save updated class names
        with open('class_names.json', 'w') as f:
            json.dump(sorted_class_names, f, indent=4)

        st.write("Updated class_names.json")
    except Exception as e:
        st.error(f"Error updating class_names.json: {e}")


def create_folders(class_name):
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

def augment_image(image, count=30):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    augmented_images = []

    for _ in range(count):
        augmented_image = datagen.flow(image, batch_size=1)[0]
        augmented_image = array_to_img(augmented_image[0])
        augmented_images.append(augmented_image)
    
    return augmented_images

def save_images(images, class_name):
    train_folder = os.path.join(train_dir, class_name)
    valid_folder = os.path.join(valid_dir, class_name)
    test_folder = os.path.join(test_dir, class_name)
    
    for i, img in enumerate(images):
        if i < 3:
            img.save(os.path.join(test_folder, f'image_{i}.jpg'))
        elif i < 9:
            img.save(os.path.join(valid_folder, f'image_{i}.jpg'))
        else:
            img.save(os.path.join(train_folder, f'image_{i}.jpg'))

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def retrain_model():
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=5,
        validation_data=valid_generator
    )

    model.save('staff_mobilenet_v2_model.h5')

def run():
    st.title("Staff Image Recognition")

    st.write("Upload an image of a staff member to get predictions and add new images to the dataset.")

    class_name = st.text_input("Enter the class name for the new images:")
    drink_preference = st.text_input("Enter drink preference for the new staff member:")
    dietary_restrictions = st.text_input("Enter dietary restrictions for the new staff member:")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None and class_name:
        create_folders(class_name)

        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        augmented_images = augment_image(image)
        save_images(augmented_images, class_name)

        update_class_names(class_name, drink_preference, dietary_restrictions)

        retrain_model()

        model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        
        # Use the predicted class index to find the name
        predicted_class_name = list(json.load(open('class_names.json')).keys())[predicted_class_index]

        st.write(f"Prediction: {predicted_class_name} with probability {np.max(predictions):.2f}")
        st.write(f"Drink Preference: {json.load(open('class_names.json')).get(predicted_class_name, {}).get('drink_preference', 'N/A')}")
        st.write(f"Dietary Restrictions: {json.load(open('class_names.json')).get(predicted_class_name, {}).get('dietary_restrictions', 'N/A')}")
