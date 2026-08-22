import streamlit as st

def apply_dark_theme():
    st.markdown("""
        <style>
            /* 1. Global Background & Text */
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                background-color: #0E1117 !important;
                color: #C9D1D9 !important;
            }
            
            [data-testid="stMain"] {
                background-color: #0E1117 !important;
            }

            /* 2. Sidebar Styling & Contrast Fixes */
            section[data-testid="stSidebar"] {
                background-color: #161B22 !important;
                border-right: 1px solid #30363D;
            }

            /* Sidebar Nav Links & Headers */
            section[data-testid="stSidebar"] *, 
            [data-testid="stSidebarNav"] a, 
            [data-testid="stSidebarNav"] span,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p {
                color: #E6EDF3 !important;
                font-weight: 500;
            }

            /* Sidebar Selected Active Link Highlight */
            [data-testid="stSidebarNav"] a[aria-current="page"] {
                background-color: #21262D !important;
                border-radius: 6px;
            }

            /* Sidebar Radio Button Options */
            div[data-testid="stMarkdownContainer"] p {
                color: #E6EDF3 !important;
            }

            /* Sidebar Info / Alert Boxes */
            [data-testid="stSidebar"] [data-testid="stAlert"] {
                background-color: #21262D !important;
                border: 1px solid #30363D !important;
                color: #58A6FF !important;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"] p {
                color: #58A6FF !important;
            }

            /* 3. Main Content Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #F0F6FC !important;
            }

            /* 4. Custom Cards for Metrics */
            .metric-card {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 14px 18px;
                margin-bottom: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
            }
            .metric-header {
                font-size: 0.88rem;
                color: #8B949E;
                font-weight: 600;
            }
            .metric-value {
                font-size: 1.4rem;
                font-weight: 700;
                color: #00D1B2;
                margin: 4px 0;
            }
            .signal-buy { color: #3FB950; font-weight: 600; font-size: 0.85rem; }
            .signal-sell { color: #F85149; font-weight: 600; font-size: 0.85rem; }
            .signal-hold { color: #8B949E; font-weight: 600; font-size: 0.85rem; }

            /* 5. Dataframe Table Dark Overrides */
            [data-testid="stDataFrame"] {
                border: 1px solid #30363D;
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)