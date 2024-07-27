import streamlit as st
import pickle

# Load the remittance prediction model
model_path = 'models/clf_model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))

# Load the label Encoder
encoder_path = 'models/label_encoder.pkl'
encoder = pickle.load(open(encoder_path, 'rb'))

def malaria_prediction(feature_inputs):
    prediction = loaded_model.predict([feature_inputs])
    predicted_country = encoder.inverse_transform([prediction])[0]
    return predicted_country

def main():
    # Streamlit app
    st.title("Malaria Hotspot Prediction by Country")

    st.write("""
    # Predict the likelihood of malaria hotspots based on various factors.
    Enter the features below:
    """)

    # Creating input fields for the features
    rural_population = st.number_input("Rural population (% of total population)", min_value=0.0, max_value=100.0, value=50.0)
    rural_population_growth = st.number_input("Rural population growth (annual %)", min_value=-10.0, max_value=10.0, value=0.0)
    urban_population = st.number_input("Urban population (% of total population)", min_value=0.0, max_value=100.0, value=50.0)
    urban_population_growth = st.number_input("Urban population growth (annual %)", min_value=-10.0, max_value=10.0, value=0.0)
    basic_drinking_water_services = st.number_input("People using at least basic drinking water services (% of population)", min_value=0.0, max_value=100.0, value=80.0)
    basic_drinking_water_services_rural = st.number_input("People using at least basic drinking water services, rural (% of rural population)", min_value=0.0, max_value=100.0, value=70.0)
    basic_drinking_water_services_urban = st.number_input("People using at least basic drinking water services, urban (% of urban population)", min_value=0.0, max_value=100.0, value=90.0)
    basic_sanitation_services = st.number_input("People using at least basic sanitation services (% of population)", min_value=0.0, max_value=100.0, value=60.0)
    basic_sanitation_services_rural = st.number_input("People using at least basic sanitation services, rural (% of rural population)", min_value=0.0, max_value=100.0, value=50.0)
    basic_sanitation_services_urban = st.number_input("People using at least basic sanitation services, urban (% of urban population)", min_value=0.0, max_value=100.0, value=70.0)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    malaria_incidence = st.number_input("Incidence of malaria (per 1,000 population at risk)", min_value=0.0, max_value=1000.0, value=10.0)

    features = [
        rural_population,
        rural_population_growth,
        urban_population,
        urban_population_growth,
        basic_drinking_water_services,
        basic_drinking_water_services_rural,
        basic_drinking_water_services_urban,
        basic_sanitation_services,
        basic_sanitation_services_rural,
        basic_sanitation_services_urban,
        year,
        malaria_incidence
    ]

    # Code for Prediction
    if st.button('Predict Malaria Incidence'):
        try:
            prediction = malaria_prediction(features)
            st.success(f'The country where this incidence of malaria is likely to have occurred is {prediction}')
        except ValueError as e:
            st.error(f'Error: {e}. Please ensure the country name is correct and matches the training data.')

if __name__ == '__main__':
    main()
