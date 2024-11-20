import os
import streamlit as st
import google.generativeai as genai


# Configure the API key
f = open("keys/geminikey.txt")
key = f.read()

genai.configure(api_key=key)

# Initialize the Generative AI model
model = genai.GenerativeModel("gemini-1.5-flash")

# System prompt for the AI
sys_prompt = """
You are an AI Code Reviewer, an expert in Python code. Review and analyze submitted code to provide:
1. ## 🪲Bug Report: Identify potential bugs, syntax errors, and logical flaws, with explanations.
2. ## ⚙️Fixed Code: Suggest corrections or optimizations with explanations.
3. ## 📄Instruction: Offer helpful,understandable and concise feedback for developers at all skill levels.
Keep the tone professional, clear, and focused on improving coding practices.
"""

# Function to get AI response
def get_response(code_input):
    response = model.generate_content([sys_prompt, code_input])
    return response.text

# Streamlit app configuration
st.set_page_config(page_title="AI Code Reviewer", page_icon="🤖", layout="wide")

# App header
st.markdown("<h1 style='text-align: center;'>🤖 AI Code Reviewer</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Analyze and optimize your code with AI</h6>", unsafe_allow_html=True)
st.markdown("---")

# Intro section
st.write("### Submit your Python code to identify issues, apply fixes, and uncover valuable suggestions!")
st.markdown("Enter your Python code below, and let the AI provide a detailed analysis and optimization!")

# Code input section
st.write("#### 📝 Code Input")
code_input = st.text_area("Enter your Python code here:", placeholder="Enter your code")

# Review button and output

if st.button("Review Code") and code_input.strip():
    st.header("🔍Code Review ")
    response = get_response(code_input)
    st.write(response)

