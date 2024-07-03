import streamlit as st
import pickle
import pandas as pd

def model_prediction(features):
    pickled_model = pickle.load(open('Notebook/Click_through_prediction_RandomForest.pkl', 'rb'))
    Clicked = pickled_model.predict(features)
    return f'Clicked on Ad: {Clicked[0]}'

# Streamlit app
def main():
    st.title("Click-Through Rate Prediction")

    st.header("Enter User Details")

    Daily_Time_Spent_on_Site = st.number_input('Daily Time Spent on Site', min_value=0, max_value=100, value=30)
    Age = st.number_input('Age', min_value=0, max_value=100, value=25)
    Daily_Internet_Usage = st.number_input('Daily Internet Usage', min_value=0, max_value=100, value=50)
    Gender = st.selectbox('Gender', [0, 1])  # Assuming 0 for Female and 1 for Male
    Area_income = st.number_input('Area Income', min_value=0, max_value=100000, value=50000)

    features = [[Daily_Time_Spent_on_Site, Age, Daily_Internet_Usage, Gender, Area_income]]

    if st.button("Predict"):
        result = model_prediction(features)
        st.subheader("Prediction")
        st.write(result)

if __name__ == "__main__":
    main()
