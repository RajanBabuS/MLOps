import os
import streamlit as st
import pandas as pd
import joblib

# Set clean minimalist page layout
st.set_page_config(page_title="Visit with Us - Package Prediction", layout="wide")

# Load the model committed by the pipeline (sits next to this file)
# Make sure your pipeline exports the entire Pipeline object containing the ColumnTransformer!
model_path = os.path.join(os.path.dirname(__file__), "wellness_package_predictor_best_model_v1.joblib")

@st.cache_resource
def load_production_model(path):
    return joblib.load(path)

try:
    model = load_production_model(model_path)
except Exception as e:
    st.error(f"Could not load model file at {model_path}. Make sure it is committed via GitHub Actions.")
    st.stop()

st.title("🏨 Wellness Tourism Package Purchase Predictor")
st.write("""
This MLOps inference app predicts whether a lead will purchase the Wellness Tourism Package *before* sales contacts them.
""")

st.markdown("---")

# Use columns layout for a compact, neat, minimalistic user dashboard interface
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=22000.0, step=500.0)

with col2:
    st.subheader("✈️ Travel Habits")
    # For UI we show Yes/No, but transform to 1/0 for raw data format tracking consistency
    passport_ui = st.selectbox("Has Passport?", ["No", "Yes"])
    passport = 1 if passport_ui == "Yes" else 0
    
    own_car_ui = st.selectbox("Owns a Car?", ["No", "Yes"])
    own_car = 1 if own_car_ui == "Yes" else 0
    
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3, step=1)
    
    # Left as raw numerical values to trigger auto-conversion pass-through mapping
    city_tier = st.selectbox("City Tier", [1, 2, 3])

with col3:
    st.subheader("📞 Engagement Details")
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    duration_pitch = st.number_input("Duration of Pitch (Min)", min_value=0, max_value=120, value=14, step=1)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=4, step=1)
    property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=3, step=1)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=1, step=1)
    satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

st.markdown("---")

# Convert the structural schema format variables to match the model features list exactly
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# Center aligned operational runtime execution button
_, btn_col, _ = st.columns([1, 1, 1])

with btn_col:
    predict_clicked = st.button("🔮 Evaluate Customer Intent", use_container_width=True)

if predict_clicked:
    # Run the model pipeline (Transformer tracks and modifies the categories array internally)
    prediction = model.predict(input_data)[0]
    
    st.subheader("Target Classification Strategy:")
    if prediction == 1:
        st.success("🎯 **Highly Likely to Purchase!** Send marketing pitch for the Wellness Package.")
    else:
        st.info("🛑 **Unlikely to Purchase.** Hold off on package pitch to save marketing overhead costs.")
