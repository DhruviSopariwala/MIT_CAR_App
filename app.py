import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #667eea;
        letter-spacing: 0.5px;
    }
    
    /* Input Containers */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Success Box - Enhanced */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(132, 250, 176, 0.3);
        border: none;
    }
    
    .price-prediction {
        font-size: 3rem;
        font-weight: 800;
        color: #2d3436;
        text-align: center;
        margin: 1rem 0;
        letter-spacing: -1px;
    }
    
    .rupee-amount {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #999;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-top: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar Styling */
    .sidebar-content {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* General Improvements */
    h3 {
        color: #667eea;
        font-weight: 700;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 10px !important;
    }
    
    /* Columns and Containers */
    .column-header {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load pickle files
final_model = pickle.load(open('final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_info = pickle.load(open('model_info.pkl', 'rb'))

# Encoding dicts from model_info
d1 = model_info['encoding_dicts']['insurance_validity']
d2 = model_info['encoding_dicts']['fuel_type']
d3 = model_info['encoding_dicts']['ownsership']
d4 = model_info['encoding_dicts']['transmission']

scaling_required = model_info['scaling_required']
model_name = model_info['model_name']

# Header Section
st.markdown('<div class="main-header">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Accurate price estimation using advanced {model_name} ML model</div>', unsafe_allow_html=True)

st.markdown("---")

# Model Information in Sidebar
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><h2 style="color: #667eea;">📊 Model Info</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">0.852</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">2.63L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">MAE</div>
            <div class="metric-value">1.70L</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-content">
        <h4 style="color: #667eea; margin-top: 0;">Model Details</h4>
        <p><strong>Type:</strong> {model_name}</p>
        <p><strong>Features:</strong> 11 parameters</p>
        <p><strong>Status:</strong> ✅ Ready to predict</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-content">
        <h4 style="color: #667eea; margin-top: 0;">💡 Prediction Tips</h4>
        <ul style="font-size: 0.95rem; line-height: 1.8;">
            <li>✅ Be accurate with specifications</li>
            <li>✅ KMS driven affects price heavily</li>
            <li>✅ Engine size & power matter</li>
            <li>✅ Newer cars cost more</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.markdown('<div class="section-header">📝 Vehicle Classification</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    insurance_validity = st.selectbox(
        '🛡️ Insurance',
        list(d1.keys()),
        help="Type of insurance coverage on the vehicle"
    )

with col2:
    fuel_type = st.selectbox(
        '⛽ Fuel',
        list(d2.keys()),
        help="Primary fuel used by the vehicle"
    )

with col3:
    transmission = st.selectbox(
        '⚙️ Transmission',
        list(d4.keys()),
        help="Type of transmission system"
    )

with col4:
    ownership = st.selectbox(
        '👤 Ownership',
        list(d3.keys()),
        help="Number of previous owners"
    )

# Numerical Features
st.markdown('<div class="section-header">🔧 Technical Specifications</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    kms_driven = st.slider(
        '🛣️ Kilometers Driven',
        min_value=0,
        max_value=300000,
        step=1000,
        value=20000,
        help="Total kilometers driven by the vehicle"
    )

with col2:
    mileage = st.number_input(
        '📊 Mileage (kmpl)',
        min_value=0.0,
        max_value=50.0,
        value=18.5,
        step=0.1,
        help="Fuel efficiency in kilometers per liter"
    )

with col3:
    engine = st.number_input(
        '🔩 Engine Capacity (cc)',
        min_value=500,
        max_value=6000,
        value=1500,
        step=50,
        help="Engine displacement in cubic centimeters"
    )

col1, col2, col3 = st.columns(3)

with col1:
    max_power = st.number_input(
        '⚡ Maximum Power (bhp)',
        min_value=30.0,
        max_value=600.0,
        value=120.0,
        step=1.0,
        help="Maximum power output in brake horsepower"
    )

with col2:
    torque = st.number_input(
        '💪 Torque (Nm)',
        min_value=50.0,
        max_value=1000.0,
        value=180.0,
        step=5.0,
        help="Maximum torque output in Newton-meters"
    )

with col3:
    seats = st.selectbox(
        '🪑 Number of Seats',
        [2, 4, 5, 6, 7, 8, 9],
        index=2,
        help="Total seating capacity of the vehicle"
    )

col1, col2 = st.columns(2)

with col1:
    manufacturing_year = st.slider(
        '📅 Manufacturing Year',
        min_value=2000,
        max_value=2024,
        value=2019,
        help="Year the vehicle was manufactured"
    )

# Prediction Section
st.markdown("---")
st.markdown('<div class="section-header">🎯 Calculate Price</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    predict_button = st.button('🔮 Predict', use_container_width=True, type="primary")

if predict_button:
    ins = d1[insurance_validity]
    fuel = d2[fuel_type]
    own = d3[ownership]
    trans = d4[transmission]
    car_age = 2024 - manufacturing_year

    features = [[ins, fuel, kms_driven, own, trans, mileage, engine, max_power, torque, car_age, seats]]

    if scaling_required:
        features = scaler.transform(features)

    yp = final_model.predict(features)[0]
    
    # Display prediction with enhanced styling
    st.markdown(f"""
    <div class="success-box">
        <div style="text-align: center;">
            <div style="font-size: 1rem; color: #2d3436; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Estimated Car Price</div>
            <div class="price-prediction">₹ {yp:.2f} Lakhs</div>
            <div class="rupee-amount">₹ {yp * 100000:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display prediction details in cards
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Price (Lakhs)</div>
            <div class="metric-value">₹{yp:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Price (Rupees)</div>
            <div class="metric-value">₹{yp * 100000:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Vehicle Age</div>
            <div class="metric-value">{car_age} yrs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">ML Model</div>
            <div class="metric-value" style="font-size: 1rem; color: #667eea;">GB</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="margin: 0; font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">🚗 Car Price Prediction System</p>
    <p style="margin: 0; font-size: 0.85rem;">Powered by Advanced Machine Learning | Gradient Boosting Model</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #ccc;">⚠️ Disclaimer: This is an estimated price based on historical data. Actual prices may vary.</p>
</div>
""", unsafe_allow_html=True)
