import streamlit as st

main_page = st.Page("pages/main_page.py", title="Home", icon="🏠", url_path="/")
bs_page = st.Page(
    "pages/bs_page.py",
    title="Black-Scholes Calculator",
    icon="🧮",
    url_path="/black-scholes",
)
vol_surface_page = st.Page(
    "pages/vol_surface_page.py",
    title="Volatility Surface",
    icon="📊",
    url_path="/volatility-surface",
)

pg = st.navigation([main_page, bs_page, vol_surface_page])
pg.run()
