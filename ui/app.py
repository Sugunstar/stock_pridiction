import streamlit as st
import requests
import base64

API_URL = "http://localhost:5000"

st.set_page_config(page_title="CandleSense", layout="centered")

st.title("🕯️ CandleSense")
st.markdown("### Candlestick Pattern Recognition & Outcome Prediction")

st.info("**Disclaimer**: This tool provides statistical/educational information based on historical patterns. It is not financial advice. Past performance does not guarantee future results.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):").upper()

if st.button("Analyze"):
    if ticker:
        with st.spinner("Analyzing data..."):
            try:
                response = requests.get(f"{API_URL}/predict?ticker={ticker}")
                if response.status_code == 200:
                    data = response.json()
                    
                    st.subheader(f"Results for {ticker}")
                    
                    # Display Image
                    img_data = base64.b64decode(data['chart_base64'])
                    st.image(img_data, caption="Recent Window Chart")
                    
                    # Pattern Info
                    st.markdown(f"### Pattern Detected: **{data['pattern_detected'].replace('_', ' ').title()}**")
                    st.markdown(f"**Classifier Confidence:** {data['cnn_confidence']}%")
                    
                    # Outcomes
                    st.markdown("### Forward Returns (Historical Edge)")
                    for horizon, prob in data['forward_returns'].items():
                        h_days = horizon.split('d')[0]
                        st.write(f"Probability of rising over next **{h_days} days**: **{prob}%**")
                        
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error("Could not connect to the API. Ensure `flask --app api.main run` is running.")
    else:
        st.warning("Please enter a ticker.")
