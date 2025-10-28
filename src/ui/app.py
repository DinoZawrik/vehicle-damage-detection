"""
Streamlit web interface for vehicle damage detection.
"""

import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import os

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .severity-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .severity-minor { background-color: #d4edda; color: #155724; }
    .severity-moderate { background-color: #fff3cd; color: #856404; }
    .severity-severe { background-color: #f8d7da; color: #721c24; }
    .severity-critical { background-color: #f5c6cb; color: #491217; }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_and_analyze(uploaded_file):
    """Upload image and get analysis results."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_URL}/api/analyze", files=files, timeout=60)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The image might be too large or the server is busy.")
        return None
    except Exception as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None


def get_history():
    """Fetch analysis history from API."""
    try:
        response = requests.get(f"{API_URL}/api/history?limit=10", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def render_severity_badge(severity):
    """Render severity level with appropriate styling."""
    if not severity:
        return "Unknown"

    severity_class = f"severity-{severity.lower()}"
    return f'<div class="{severity_class} severity-box"><h3>{severity.upper()}</h3></div>'


def main():
    # Header
    st.markdown('<div class="main-header">Vehicle Damage Detection System</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150?text=Vehicle+AI", use_container_width=True)
        st.markdown("### About")
        st.info(
            "This system uses advanced computer vision to detect and assess vehicle damage. "
            "Upload an image to get instant analysis including severity assessment and cost estimation."
        )

        # API status
        api_status = check_api_health()
        status_color = "green" if api_status else "red"
        status_text = "Online" if api_status else "Offline"
        st.markdown(f"**API Status:** :{status_color}[{status_text}]")

    # Main content tabs
    tab1, tab2 = st.tabs(["Analyze Image", "History"])

    with tab1:
        st.markdown("### Upload Vehicle Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the vehicle for analysis"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            if st.button("Analyze Damage", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... This may take a few seconds."):
                    results = upload_and_analyze(uploaded_file)

                    if results:
                        with col2:
                            st.markdown("#### Analysis Results")

                            # Severity
                            severity = results.get("classification", {}).get("severity")
                            st.markdown(render_severity_badge(severity), unsafe_allow_html=True)

                            # Key metrics
                            metrics_col1, metrics_col2 = st.columns(2)

                            with metrics_col1:
                                damage_count = results.get("classification", {}).get("damage_count", 0)
                                st.metric("Damages Detected", damage_count)

                                confidence = results.get("classification", {}).get("avg_confidence", 0)
                                st.metric("Avg Confidence", f"{confidence*100:.1f}%")

                            with metrics_col2:
                                cost = results.get("cost_estimate", {}).get("estimated_cost", 0)
                                currency = results.get("cost_estimate", {}).get("currency", "USD")
                                st.metric("Estimated Cost", f"{cost:,.0f} {currency}")

                                area_ratio = results.get("classification", {}).get("area_ratio", 0)
                                st.metric("Damage Coverage", f"{area_ratio*100:.1f}%")

                        # Detailed results
                        with st.expander("View Detailed Results"):
                            st.markdown("**Damage Types:**")
                            damage_types = results.get("classification", {}).get("damage_types", {})
                            if damage_types:
                                df = pd.DataFrame(list(damage_types.items()), columns=["Type", "Count"])
                                st.dataframe(df, use_container_width=True)

                            st.markdown("**Cost Breakdown:**")
                            cost_data = results.get("cost_estimate", {})
                            st.write(f"- Parts: {cost_data.get('parts_cost', 0):,.0f} {currency}")
                            st.write(f"- Labor: {cost_data.get('labor_cost', 0):,.0f} {currency}")
                            st.write(f"- Range: {cost_data.get('min_cost', 0):,.0f} - {cost_data.get('max_cost', 0):,.0f} {currency}")

                            st.markdown("**Processing Time:**")
                            proc_time = results.get("total_processing_time", 0)
                            st.write(f"{proc_time:.2f} seconds")

                        st.success("Analysis completed successfully!")

    with tab2:
        st.markdown("### Analysis History")

        if st.button("Refresh History"):
            st.rerun()

        history = get_history()

        if history:
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

            # Display table
            st.dataframe(
                df[["id", "image_filename", "severity", "damage_count", "estimated_cost", "created_at"]],
                use_container_width=True,
                column_config={
                    "id": "ID",
                    "image_filename": "Image",
                    "severity": "Severity",
                    "damage_count": "Damages",
                    "estimated_cost": st.column_config.NumberColumn("Cost (USD)", format="$%.2f"),
                    "created_at": "Date"
                }
            )
        else:
            st.info("No analysis history available yet.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Vehicle Damage Detection System | Powered by YOLOv8 & FastAPI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
