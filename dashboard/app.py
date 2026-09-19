import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
st.set_page_config(page_title="SOC Monitor", layout="wide", page_icon="🛡️")

# CSS for dark SOC theme
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI-Powered Network Security Monitor")

# Fetch Stats
def get_stats():
    try:
        res = requests.get(f"{API_BASE_URL}/stats")
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return {"total_alerts": 0, "high_critical_alerts": 0, "anomalies_detected": 0}

stats = get_stats()

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Traffic Processed", "50,000", "+1,200/hr") # Placeholder for traffic volume
col2.metric("Total Security Alerts", stats["total_alerts"])
col3.metric("High/Critical Alerts", stats["high_critical_alerts"])
col4.metric("Unseen Anomalies", stats["anomalies_detected"])

st.markdown("---")

# Fetch Alerts
@st.cache_data(ttl=5)
def get_alerts():
    try:
        res = requests.get(f"{API_BASE_URL}/alerts?limit=500")
        if res.status_code == 200:
            return pd.DataFrame(res.json())
    except:
        pass
    return pd.DataFrame()

df_alerts = get_alerts()

if not df_alerts.empty:
    df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
    
    col_charts1, col_charts2 = st.columns(2)
    
    with col_charts1:
        st.subheader("Alerts over Time")
        alerts_time = df_alerts.groupby(df_alerts['timestamp'].dt.floor('H')).size().reset_index(name='count')
        if not alerts_time.empty:
            fig = px.line(alerts_time, x='timestamp', y='count', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
    with col_charts2:
        st.subheader("Severity Distribution")
        fig2 = px.pie(df_alerts, names='severity', hole=0.4, template='plotly_dark',
                      color='severity', 
                      color_discrete_map={'CRITICAL':'red', 'HIGH':'orange', 'MEDIUM':'yellow', 'LOW':'blue', 'INFO':'green'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Recent Security Events")
    
    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    sev_filter = f_col1.multiselect("Severity", options=df_alerts['severity'].unique(), default=df_alerts['severity'].unique())
    pred_filter = f_col2.multiselect("Prediction", options=df_alerts['prediction'].unique(), default=df_alerts['prediction'].unique())
    
    filtered_df = df_alerts[
        (df_alerts['severity'].isin(sev_filter)) & 
        (df_alerts['prediction'].isin(pred_filter))
    ]
    
    st.dataframe(
        filtered_df[['timestamp', 'source_ip', 'destination_ip', 'prediction', 'confidence', 'severity']],
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("Alert Details & SHAP Explanations")
    selected_alert_id = st.selectbox("Select Alert ID to investigate:", filtered_df['id'].tolist() if not filtered_df.empty else [])
    
    if selected_alert_id:
        try:
            res = requests.get(f"{API_BASE_URL}/alerts/{selected_alert_id}")
            if res.status_code == 200:
                alert_detail = res.json()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Threat Info**")
                    st.json({
                        "Prediction": alert_detail['prediction'],
                        "Severity": alert_detail['severity'],
                        "Confidence": round(alert_detail['confidence'], 4),
                        "Anomaly Score": round(alert_detail['anomaly_score'], 4),
                        "Source": alert_detail['source_ip'],
                        "Destination": alert_detail['destination_ip']
                    })
                    
                with c2:
                    st.write("**AI Explanation (SHAP)**")
                    if alert_detail.get('explanation'):
                        st.write("Top contributing features:")
                        shap_df = pd.DataFrame(alert_detail['explanation']['top_features'])
                        st.dataframe(shap_df[['feature', 'original_value', 'shap_value']], use_container_width=True)
                    else:
                        st.info("No detailed explanation available for this alert.")
        except Exception as e:
            st.error(f"Could not load details: {e}")
else:
    st.info("No alerts found in the database. Send traffic to the API to populate the dashboard.")
