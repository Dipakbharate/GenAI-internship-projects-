# 🤖👁️ SenseAI – AI-Powered Assistive System

## 🚀 Overview
**SenseAI** is an AI-driven assistive application designed to enhance the independence and daily life of visually impaired individuals.  

The system leverages **Computer Vision, OCR, and Generative AI (LLMs)** to provide real-time scene understanding, object detection, and audio-based feedback, enabling users to interact with their surroundings confidently.

---

## 💡 Problem Statement
Visually impaired individuals often face challenges in understanding their environment, reading text, and navigating safely.  

SenseAI addresses this by providing **real-time AI assistance** through image analysis, speech output, and intelligent responses.

---

## 🧠 Key Features

### 🔍 Real-Time Scene Understanding
- Generates contextual image descriptions using **Google Generative AI (Gemini-1.5 Flash)**
- Converts descriptions into audio using **pyttsx3**

---

### 🗣️ Text-to-Speech from Visual Content
- Extracts text from images using **pytesseract (OCR)**
- Reads content aloud for accessibility
- Supports **offline audio playback**

---

### 🚧 Object & Obstacle Detection
- Detects objects using **YOLOv8**
- Provides both **visual and textual outputs**
- Helps users understand surroundings in real time

---

### 🛠️ Personalized Assistance
- Combines image input + user queries
- Uses **LLMs** to generate intelligent responses
- Outputs in both **text and speech formats**

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frontend:** Streamlit  
- **Computer Vision:** YOLOv8, OpenCV  
- **OCR:** pytesseract  
- **Text-to-Speech:** pyttsx3  
- **Generative AI:** Google Gemini API  
- **Libraries:** Pandas, NumPy  

---

## ⚙️ How It Works

1. 📥 **Input**
   - User uploads an image via Streamlit interface  

2. 🧠 **Processing**
   - Scene understanding using Gemini API  
   - Object detection using YOLOv8  
   - Text extraction using OCR  

3. 🔊 **Output**
   - AI-generated description  
   - Detected objects  
   - Extracted text  
   - Audio feedback (TTS)  

---

## 🎯 Use Cases

- 🖼️ **Scene Description** – Understand surroundings through AI-generated captions  
- 📖 **Text Reading** – Read menus, documents, and labels  
- 🚶 **Navigation Assistance** – Detect obstacles and objects  
- 🤖 **Personalized Help** – Get AI-powered answers to queries  

---

## 🌱 Future Enhancements
Voice command integration for hands-free interaction.
Mobile app deployment for better portability.
Real-time video analysis for dynamic obstacle detection.
Multilingual support in text-to-speech and OCR features.

## 🤝 Acknowledgments
I express my heartfelt gratitude to Innomatics Research Labs and Kanav Bansal for their guidance and support throughout this project.

