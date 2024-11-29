import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import google.generativeai as genai
from ultralytics import YOLO
import io
import threading

# Set Google Generative AI API Key
f = open("keys/geminikey.txt")
key = f.read()
genai.configure(api_key=key)

# Initialize Text-to-Speech (using pyttsx3)
engine = pyttsx3.init()

# Initialize YOLO model for object detection
model = YOLO("yolov8n.pt")  # Use YOLOv8 model weights

# Image to bytes function for uploading image
def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Global variable to track speech state
is_speaking = False
speech_thread = None

# Text-to-Speech function using pyttsx3
def text_to_speech_pyttsx3(text):
    global is_speaking, speech_thread
    
    def speak():
        global is_speaking
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

    # If speech is already playing, do nothing
    if not is_speaking:
        speech_thread = threading.Thread(target=speak)
        speech_thread.start()

# Stop the speech
def stop_speech():
    global is_speaking
    is_speaking = False
    engine.stop()

# Function to generate response from the model (Google Generative AI)
def get_response(input_prompt, image_data):
    try:
        # Convert the image data into a PIL Image object
        image_data_pil = Image.open(io.BytesIO(image_data))  # Convert bytes to PIL Image
        
        # Create the model object and generate content
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([input_prompt, image_data_pil])  # Pass prompt and image
        return response.text
    
    except Exception as e:
        st.error(f"Error with Google Generative AI: {e}")
        return "⚠️ Sorry, I couldn't generate a description for this image."

# Streamlit Application Interface
st.set_page_config(page_title="SenseAI🤖", layout="wide")
st.title("🌟 SenseAI: Bridging Vision with AI-Powered Assistance 🤖👁️")
st.sidebar.header("⚙️ Features")
features = st.sidebar.multiselect(
    "Select features to activate: 🌐",
    [
        "🔍 Real-Time Scene Understanding",
        "🗣️ Text-to-Speech Conversion for Visual Content",
        "🚧 Object and Obstacle Detection",
        "🛠️ Personalized Assistance for Daily Tasks"
    ]
)

# Create two columns: one for uploading and displaying the image, and another for the remaining features
col1, col2 = st.columns([1, 2])

# Column 1: Upload Image Section
with col1:
    st.markdown("### 📤 Upload and Display Image")
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        # Display uploaded image with a caption
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

# Column 2: Main Functionalities
with col2:
    st.markdown("### 🛠️ Image Analysis and Assistance")

    # Real-Time Scene Understanding
    if uploaded_file and "🔍 Real-Time Scene Understanding" in features:
        st.subheader("🔍 Scene Understanding")
        input_prompt = """
        You are an AI assistant designed to assist visually impaired individuals
        by analyzing images and providing descriptive outputs.
        Your task is to:
        - Analyze the uploaded image and describe its content in clear and simple language.
        - Provide detailed information about objects, people, settings, or activities in the scene
        """
        
        # Convert image to bytes
        image_data = image_to_bytes(image)

        # Generate description using Google Generative AI
        response = get_response(input_prompt, image_data)
        st.write("📝 Description:", response)

        # Store the response in session state for future use
        st.session_state.image_description = response

        # Automatically play the audio for the scene description
        text_to_speech_pyttsx3(response)

    # Text-to-Speech Conversion (OCR)
    if uploaded_file and "🗣️ Text-to-Speech Conversion for Visual Content" in features:
        st.subheader("🗣️ Text-to-Speech Conversion")
        extracted_text = pytesseract.image_to_string(image)
        st.write("📝 Extracted Text:", extracted_text)

        # Automatically play the audio for the extracted text
        text_to_speech_pyttsx3(extracted_text)

    # Object and Obstacle Detection
    if uploaded_file and "🚧 Object and Obstacle Detection" in features:
        st.subheader("🚧 Object and Obstacle Detection")
        results = model(image)

        # Render results on the uploaded image
        detected_image = results[0].plot()  # YOLO plotting method
        st.image(detected_image, caption="📌 Detected Objects", use_column_width=True)

        # Display detected objects
        detected_objects = []
        for result in results[0].boxes.data.cpu().numpy():  # Each box in the result
            class_idx = int(result[5])  # The class index is typically stored at position 5
            detected_objects.append(model.names[class_idx])  # `model.names` holds the class names
        
        st.write("📝 Detected Objects:", ", ".join(detected_objects))

    # Personalized Assistance
    if uploaded_file and "🛠️ Personalized Assistance for Daily Tasks" in features:
        st.subheader("🛠️ Personalized Assistance")
        user_query = st.text_input("💬 Describe what assistance you need:")
        if user_query:
            try:
                # Retrieve the stored image description from session state
                image_description = st.session_state.get("image_description", "No description available.")
                image_data = image_to_bytes(image)
                personalized_input_prompt = f"The uploaded image shows: {image_description}. Help with: {user_query}"
                assistance_response = get_response(personalized_input_prompt, image_data)
                st.write("📝 Assistance Response:", assistance_response)
                text_to_speech_pyttsx3(assistance_response)
            except Exception as e:
                st.error(f"⚠️ Error with personalized assistance: {e}")

    # Stop Audio button
    if st.button("❌ Stop Audio"):
        stop_speech()
        st.write("🔇 Audio stopped.")
