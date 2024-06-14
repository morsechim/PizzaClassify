import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Set page title and icon
st.set_page_config(page_title="Pizza Classification", page_icon="🍕")

# Function to load YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        model.fuse()  # Prepare model for inference
        return model
    except Exception as e:
        st.error("Error loading model.")
        st.error(f"Exception: {e}")
        st.stop()

# Main function
def main():
    st.title("🍕 Pizza Classification")
    st.sidebar.title("Settings")

    # Load YOLO model
    model_name = "pizza_classify.pt"
    pizza_model_path = os.path.join(".", "weights", model_name)
    pizza_classify = load_model(pizza_model_path)

    # Upload image and set confidence threshold
    uploaded_image = st.sidebar.file_uploader("Upload an image of a pizza", type=["jpg", "jpeg", "png"])
    confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.05)

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='✅ Uploaded pizza image.', use_column_width=True)

        # Classify image and display results
        with st.spinner('🤹🏽‍♀️ Classifying...'):
            try:
                results = pizza_classify.predict(image)

                any_class_above_threshold = False
                results_displayed = False

                for result in results:
                    probs = result.probs.data.tolist()
                    names = result.names

                    # Filter results based on confidence threshold
                    filtered_results = [(names[i], probs[i]) for i in range(len(names)) if probs[i] >= confidence_threshold]

                    if filtered_results:
                        any_class_above_threshold = True
                        if not results_displayed:
                            st.subheader("🧑‍🍳 This is a pizza classes:")
                            results_displayed = True
                        df_results = pd.DataFrame(filtered_results, columns=['Class', 'Confidence'])
                        st.dataframe(df_results.style.format({'Confidence': '{:.2f}'}))

                if not any_class_above_threshold:
                    st.warning(f"No classes found above confidence threshold ({confidence_threshold}).")

                st.success('🎉 Classification successful!')

            except Exception as e:
                st.error('❌ Error classifying.')
                st.error(e)

    else:
        st.markdown('Please upload an image of a pizza.')

if __name__ == "__main__":
    main()