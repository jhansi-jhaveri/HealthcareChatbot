# Healthcare Disease Prediction Chatbot

## Description
This project predicts diseases based on selected symptoms using machine learning.  
It also provides recommended precautions and medications.

## Dataset
- Training.csv and Testing.csv are used for model training and evaluation.  
- Each row contains symptoms (features) and the disease label (`prognosis`).

## Installation

1. Clone the repository:

git clone https://github.com/jhansi-jhaveri/HealthcareChatbot.git  
cd HealthcareChatbot  

2. Install dependencies:

pip install streamlit pandas scikit-learn matplotlib numpy  

3. Run the Streamlit app:

streamlit run app.py  

This will open a browser window (usually at http://localhost:8501) where you can:

- Select your symptoms  
- See the predicted disease  
- Get recommended precautions and medications  

## Datasets
- Training.csv → Used to train the model  
- Testing.csv → Used to evaluate accuracy  
Each row contains symptoms as features and the disease label (`prognosis`).

## License

This project is intended **solely for educational and academic purposes**.  
It demonstrates machine learning concepts applied to healthcare data and **does not provide medical advice**.  
Users should **not rely on this application for real-world medical decisions**.
