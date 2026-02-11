import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="IndoorFarmAI", layout="wide")
st.title("🌾 **IndoorFarmAI** - Smart Crop Advisor")
st.markdown("**Crop Recommendation + Profit Calculator**")

# Load trained model
@st.cache_data
def load_model():
    np.random.seed(42)
    crops = ['lettuce', 'spinach', 'tomato', 'basil', 'kale']
    data = []
    for crop in crops:
        if crop in ['lettuce', 'spinach', 'kale']:  # Leafy greens
            n,p,k,t,h,ph,r = 30,25,40,22,75,6.2,90
        else:  # Fruiting crops
            n,p,k,t,h,ph,r = 50,40,60,28,65,6.0,80
        for _ in range(500):
            data.append([n+np.random.normal(0,8), p+np.random.normal(0,6), k+np.random.normal(0,10),
                        t+np.random.normal(0,3), h+np.random.normal(0,10), ph+np.random.normal(0,0.3), r+np.random.normal(0,20), crop])
    
    df = pd.DataF
