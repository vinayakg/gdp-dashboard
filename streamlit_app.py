import streamlit as st
import os
from report_processor import ReportComparator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Bank Report Comparator", layout="wide")

st.title("🏦 AI Bank Report Comparison Tool")
st.write("""
This tool performs comprehensive analysis of bank annual reports using AI.
Enter the URLs of PDF reports below to get started.
""")

with st.sidebar:
    st.header("Configuration")
    
    # Client bank information
    st.subheader("Client Bank")
    client_name = st.text_input("Client Bank Name", value="ADIB")
    client_url = st.text_input("Client Bank Report URL", value="https://www.adib.com/en/siteassets/annual%20reports/integrated-annual-report_2024_en.pdf")
    
    # Other banks information
    st.subheader("Other Banks (Maximum 2)")
    
    # First bank
    st.markdown("#### First Bank")
    comp1_name = st.text_input("Name", key="comp1_name", value="FAB")
    comp1_url = st.text_input("Report URL", key="comp1_url", value="https://www.bankfab.com/-/media/fab-uds/about-fab/investor-relations/reports-and-presentations/quarterly-and-annual-reports/2024/fab-annual-report-2024-en.pdf?view=1")
    
    # Second bank
    st.markdown("#### Second Bank")
    comp2_name = st.text_input("Name", key="comp2_name", value="DIB")
    comp2_url = st.text_input("Report URL", key="comp2_url", value="https://www.dib.ae/docs/default-source/reports/dib-integrated-annual-report-2024.pdf")
    
    # Analysis depth settings
    st.subheader("Analysis Settings")
    deep_analysis = st.checkbox("Enable Deep Analysis", value=True, 
                               help="Performs more comprehensive analysis with second-level interpretation")

# Prepare competitor lists
competitor_names = []
competitor_urls = []

if comp1_name and comp1_url:
    competitor_names.append(comp1_name)
    competitor_urls.append(comp1_url)

if comp2_name and comp2_url:
    competitor_names.append(comp2_name)
    competitor_urls.append(comp2_url)

# Display current selection
st.subheader("Current Selection")
st.write(f"**Client:** {client_name}")
st.write(f"**Client URL:** {client_url}")

if competitor_names:
    st.write("**Other Banks:**")
    for i, name in enumerate(competitor_names):
        st.write(f"{i+1}. {name}")

# Analysis button
if st.button("Analyze Reports"):
    if not client_url or not client_name:
        st.error("Please provide client bank information")
    elif not competitor_names:
        st.error("Please provide at least one other bank")
    else:
        with st.spinner("Performing comprehensive analysis. This may take 10-15 minutes..."):
            try:
                progress_placeholder = st.empty()
                progress_placeholder.info("Initializing deep analysis...")
                
                comparator = ReportComparator()
                
                progress_placeholder.info("Downloading and processing PDFs...")
                results = comparator.compare_reports(
                    client_name=client_name,
                    client_url=client_url,
                    competitor_names=competitor_names,
                    competitor_urls=competitor_urls
                )
                
                progress_placeholder.empty()
                st.success("Analysis Complete!")
                
                # Display consolidated report without tabs
                st.markdown(results["consolidated_report"])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

