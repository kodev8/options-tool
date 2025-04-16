import streamlit as st

st.set_page_config(page_title="Options Analytics Tool", page_icon="🏠")
st.title("Options Pricing & Analysis Tool")

st.header("Welcome to the Options Analytics Suite")

st.markdown("""
This comprehensive tool helps traders, analysts, and students understand options pricing and volatility dynamics using the Black-Scholes model and market data.

### Key Features

- **Black-Scholes Calculator**: Price options and calculate Greeks (Delta, Gamma, Theta, Vega, etc.)
- **Volatility Surface Visualization**: Analyze implied volatility patterns across strikes and maturities
- **Real-time Market Data**: Connect to live market data for accurate pricing and analysis

### Getting Started

1. Use the sidebar to navigate between different tools
2. Enter your parameters (ticker, strike, expiration, etc.)
3. Analyze the results through interactive visualizations

### Available Tools

- **Black-Scholes Calculator**: Calculate theoretical option prices and Greeks
- **Volatility Surface**: Visualize and analyze the volatility surface for any ticker
- **Option Chain Analysis**: Explore the full option chain with key metrics

### Understanding the Models

The Black-Scholes model makes several assumptions:
- Efficient markets with no arbitrage opportunities
- No transaction costs or taxes
- Risk-free interest rate is constant
- Stock prices follow a lognormal distribution
- No dividends (though our implementation includes dividend adjustments)

### Interpreting Results

- **Option Price**: The theoretical fair value of the option
- **Greeks**: Sensitivity measures that help understand risk exposure
- **Implied Volatility**: Market's expectation of future volatility

### Advanced Features

- Support for different option types (stock, futures, FX)
- Volatility smile and term structure analysis
- Customizable visualization options
""")

# Add a section with example use cases
st.subheader("Example Use Cases")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Trading Strategy Analysis**
    - Evaluate option spreads and combinations
    - Assess risk/reward profiles
    - Identify mispriced options
    """)

with col2:
    st.markdown("""
    **Risk Management**
    - Calculate portfolio Greeks
    - Stress test positions
    - Monitor volatility exposure
    """)

# Add a footer with additional information
st.markdown("---")
st.caption(
    "Developed using Streamlit, Python, and financial mathematics principles. Data provided by Yahoo Finance."
)
