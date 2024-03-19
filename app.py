import pickle
import json
import numpy as np
import streamlit as st

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open(r"columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    with open(r"banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

def main():
    load_saved_artifacts()
    st.title("House Price Estimator")

    st.write("## Select House Details")
    total_sqft = st.number_input("Total Square Feet")
    location = st.selectbox("Location", get_location_names())
    bhk = st.number_input("Number of Bedrooms", min_value=1, step=1)
    bath = st.number_input("Number of Bathrooms", min_value=1, step=1)

    if st.button("Predict Price"):
        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        st.write(f"Estimated Price: {estimated_price} Lakhs")

if __name__ == '__main__':
    main()
