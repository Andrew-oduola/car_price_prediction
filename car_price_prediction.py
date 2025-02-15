import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('car_price_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict car price
def predict_car_price(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# USD to Lakh conversion factor
EXCHANGE_RATE = 75  # Example rate (1 USD = 75 INR)
INR_TO_LAKH = 100000  # Since 1 lakh = 100,000 INR

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🚗 Car Price Prediction")
    st.markdown("This app predicts the price of a car based on its specifications.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict car prices.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year (the car was bought)", min_value=1980, max_value=2025,  value=2011, help="Enter the year the car was bought")
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=5200, help="Enter the total kilometers driven by the car")
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"], help="Select the fuel type of the car")
        
    with col2:
        present_price_usd = st.number_input("Present Price (in USD)", min_value=0.0, value=30000.0, help="Enter the present price of the car in USD")
        owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"], help="Select the number of owners the car has had")
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"], help="Select the transmission type of the car")

    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"], help="Select the seller type")

    # Convert user inputs to model-compatible format
    fuel = 0 if fuel == "Petrol" else 1 if fuel == "Diesel" else 2
    owner = 0 if owner == "First Owner" else 1 if owner == "Second Owner" else 2 if owner == "Third Owner" else 3
    transmission = 0 if transmission == "Manual" else 1
    seller_type = 0 if seller_type == "Dealer" else 1

    # Convert present price from USD to lakh (for model input)
    present_price_lakh = (present_price_usd * EXCHANGE_RATE) / INR_TO_LAKH

    # Prepare input data for the model
    input_data = [year, present_price_lakh, km_driven, fuel, seller_type, transmission, owner]

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = predict_car_price(input_data)
            
            # Convert predicted price from lakh to USD
            predicted_price_usd = (prediction[0] * INR_TO_LAKH) / EXCHANGE_RATE

            st.success(f"Predicted Car Price: **${predicted_price_usd:.2f} USD**")
            result_placeholder.success(f"Predicted Car Price: **${predicted_price_usd:.2f} USD**")

            st.markdown("**Note:** This is a simplified model and may not be accurate for all cases.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()