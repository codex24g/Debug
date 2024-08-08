import streamlit as st
import json
import os

def load_class_names():
    if os.path.exists('class_names.json'):
        with open('class_names.json', 'r') as f:
            return json.load(f)
    return {}

def save_class_names(class_names):
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)

def run():
    st.title("Edit Staff Information")

    class_names = load_class_names()
    
    staff_names = list(class_names.keys())
    selected_staff = st.selectbox("Select staff member to edit:", staff_names)

    if selected_staff:
        st.write(f"Editing information for {selected_staff}")

        drink_preference = st.text_input("Drink Preference:", value=class_names[selected_staff].get("drink_preference", ""))
        dietary_restrictions = st.text_input("Dietary Restrictions:", value=class_names[selected_staff].get("dietary_restrictions", ""))

        if st.button("Save Changes"):
            class_names[selected_staff] = {
                "drink_preference": drink_preference,
                "dietary_restrictions": dietary_restrictions
            }
            save_class_names(class_names)
            st.success(f"Updated information for {selected_staff}")

    if st.button("Reload"):
        class_names = load_class_names()
        st.success("Reloaded class names successfully")
