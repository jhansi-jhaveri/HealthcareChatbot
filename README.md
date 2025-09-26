# Healthcare Chatbot for Disease Prediction

## Description

This project is an AI-powered chatbot that predicts diseases based on user-selected symptoms using Machine Learning.
It also provides top-3 predictions with probabilities, model accuracy, recommended precautions, and medications.

⚠️ Note: This project is for educational purposes only. It is not intended for real-world medical use..

## Dataset

- Training.csv → Used for training the model
- Testing.csv → Used for evaluating accuracy
- Each row contains multiple symptoms (features) and a corresponding disease label (prognosis).

## Installation

1. Clone the repository:

git clone https://github.com/jhansi-jhaveri/HealthcareChatbot.git  
cd HealthcareChatbot  

2. Install dependencies:

pip install streamlit pandas scikit-learn matplotlib numpy  

3. Run the Streamlit app:

streamlit run app.py  

This will open a browser window (usually at http://localhost:8501) where you can:

- Select multiple symptoms from the sidebar 
- Predict the most probable disease with confidence scores  
- Display top-3 predictions with probabilities
-

## Features

-Select multiple symptoms
-Predict the most probable disease
-Get recommended precautions & medications
-Interactive Streamlit web app

## Tech Stack

-Python (pandas, numpy, scikit-learn, matplotlib)
-Machine Learning (Classification model (disease prediction))
-Streamlit (Web deployment & interactive UI)

## Demo

https://healthcarechatbot-pspfxkfxxsw9mrfdhbwbe6.streamlit.app/

## License

This project is intended **solely for educational and academic purposes**.  
It demonstrates machine learning concepts applied to healthcare data and **does not provide medical advice**.  
Users should **not rely on this application for real-world medical decisions**.
