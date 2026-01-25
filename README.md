# Study Buddy – AI-Powered Student Assistance System
📌 Project Overview

Study Buddy is an AI-powered chatbot designed to help students by answering academic questions instantly. It uses machine learning and natural language processing to provide accurate responses and supports text, document, and image-based inputs.

🎯 Problem Statement

Students often struggle to get instant answers to academic doubts, especially outside classroom hours. Existing solutions may provide incorrect responses or lack support for learning materials like PDFs or images.

💡 Proposed Solution

Study Buddy provides a smart, reliable, and easy-to-use learning assistant that:

Answers student questions using machine learning

Supports file uploads for document-based learning

Avoids incorrect answers using confidence-based prediction

Offers a simple and interactive chat interface

⚙️ System Approach

User submits a question or uploads a file

Text is extracted using OCR or document parsers

TF-IDF converts text into numerical features

Logistic Regression predicts the most relevant answer

Confidence threshold checks prediction reliability

Response is displayed via Streamlit interface

🧠 Technologies Used

Programming Language: Python

Machine Learning: Logistic Regression

NLP: TF-IDF Vectorization

Web Framework: Streamlit

OCR: Tesseract

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib

Model Storage: Joblib

LLM Integration: Ollama (optional)

🚀 Features

AI-powered question answering

Confidence-based response handling

PDF, DOCX, image, and text support

Real-time chat interface

Model retraining support for scalability

📊 Results

Accurate answers for academic queries

Reduced incorrect responses using confidence threshold

Improved user interaction through file-based learning
