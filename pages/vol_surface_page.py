import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from tools.volatility_calc import get_data, get_risk_free_rate


st.set_page_config(
    page_title="Options Volatility Surface", page_icon="📈",
)

with open("pages/styles/custom.css", "r") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">Volatility Surface Analysis</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="info-text">Analyze and visualize implied volatility patterns across strikes and maturities</div>',
    unsafe_allow_html=True,
)

# Initialize variables
today = datetime.date(datetime.today())
maturity_filter_dict = {
    "2w": 14,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "9m": 270,
    "12m": 360,
    "18m": 540,
    "24m": 720,
    "30m": 900,
    "36m": 1080,
}
with st.sidebar:
    st.header("Parameters")
    st.caption("Enter the parameters for the volatility surface")
    ticker = st.text_input("Ticker", value="AAPL")
    min_date = today + timedelta(days=1)
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=today + timedelta(days=maturity_filter_dict["36m"]),
    )
    end_date = st.date_input(
        "End Date",
        value=min_date + timedelta(days=90),
        min_value=min_date + timedelta(days=maturity_filter_dict["2w"]),
        max_value=today + timedelta(days=maturity_filter_dict["36m"]),
    )
    risk_free_rate = st.number_input(
        "Risk-Free Rate",
        value=get_risk_free_rate(),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    st.header("Visualization")
    st.caption("Choose the visualization type")
    visualization_type = st.selectbox(
        "Visualization Type",
        options=[
            "Surface",
            "Moneyness",
            "Heatmap",
            "Smile Curves",
            "ATM Term Structure",
        ],
    )

    option_type_filter = st.radio(
        "Option Type", options=["Both", "Calls Only", "Puts Only"]
    )

    color_scale = st.selectbox(
        "Color Scale",
        options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
    )

    # Load data
    with st.spinner("Loading option data..."):
        data = get_data(ticker, start_date, end_date)

    # Strike filter
    if data is not None and not data.empty:
        st.header("Strike Filter")
        if option_type_filter == "Calls Only":
            data = data[data["option_type"] == "C"]
        elif option_type_filter == "Puts Only":
            data = data[data["option_type"] == "P"]

        min_strike = float(data["strike"].min())
        max_strike = float(data["strike"].max())
        strike_filter = st.slider(
            "Strike Range",
            min_value=min_strike,
            max_value=max_strike,
            value=(min_strike, max_strike),
        )
        # Moneyness range slider
        min_moneyness = float(data["moneyness"].min())
        max_moneyness = float(data["moneyness"].max())
        moneyness_filter = st.slider(
            "Moneyness Range",
            min_value=min_moneyness,
            max_value=max_moneyness,
            value=(min_moneyness, max_moneyness),
            step=0.01,
        )

        # IV range slider
        min_iv = float(data["impliedVolatility"].min())
        max_iv = float(data["impliedVolatility"].max())
        iv_filter = st.slider(
            "IV Range",
            min_value=min_iv,
            max_value=max_iv,
            value=(min_iv, max_iv),
            step=0.01,
        )

        with st.expander("View raw data sample"):
            st.write(f"Total data points: {len(data)}")
            st.write(f"Unique strikes: {len(data['strike'].unique())}")
            st.write(f"Unique maturities: {len(data['T'].unique())}")
            st.dataframe(data.head(10))

if data is None or data.empty:
    st.error("No options data found for the given ticker and date range")
    st.stop()

# Apply filters
data = data[(data["strike"] >= strike_filter[0]) & (data["strike"] <= strike_filter[1])]
data = data[
    (data["moneyness"] >= moneyness_filter[0])
    & (data["moneyness"] <= moneyness_filter[1])
]
data = data[
    (data["impliedVolatility"] >= iv_filter[0])
    & (data["impliedVolatility"] <= iv_filter[1])
]

# Clean data - remove NaN and infinite values
data = data.replace([np.inf, -np.inf], np.nan)
# data = data.dropna(subset=["impliedVolatility"])
# data = data[(data["impliedVolatility"] > 0.01) & (data["impliedVolatility"] < 2.0)]
if len(data) == 0:
    st.error("No valid data after filtering. Try adjusting your filters.")
    st.stop()

# Display summary information
with st.container():

    st.markdown(
    f"""
    <div class="highlight metric-container">
        <div class="metric-card">
            <div class="metric-value">{ticker}</div>
            <div class="metric-label">Ticker</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${data["underlying_price"].iloc[0]:.2f}</div>
            <div class="metric-label">Underlying Price</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{data["impliedVolatility"].mean():.2%}</div>
            <div class="metric-label">Average IV</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Data Points</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Visualizations
st.markdown('<div class="sub-header">Volatility Analysis</div>', unsafe_allow_html=True)

if visualization_type in ["Surface", "Moneyness"]:
    # Prepare data for surface plot
    index_col = "strike" if visualization_type == "Surface" else "moneyness"

    # Make sure we have enough data points
    if len(data[index_col].unique()) < 3 or len(data["T"].unique()) < 3:
        st.warning(
            "Not enough data points to create a surface. Try expanding your date range."
        )
    else:
        # Create a pivot table with proper handling of missing values
        pivot_data = data.pivot_table(
            values="impliedVolatility", index=index_col, columns="T", aggfunc="mean"
        )

        # Fill missing values for better visualization
        # pivot_data = pivot_data.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)

        # # Choose regular grid for interpolation
        # x_grid = np.linspace(data[index_col].min(), data[index_col].max(), 50)
        # y_grid = np.linspace(data["T"].min(), data["T"].max(), 50)
        # X, Y = np.meshgrid(x_grid, y_grid)

        pivot_data = pivot_data.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
        index_col = "strike" if visualization_type == "Surface" else "moneyness"
        x_data = data[index_col]
        y_data = data["T"]
        z_data = data["impliedVolatility"]
        x_lin = np.linspace(x_data.min(), x_data.max(), 50)
        y_lin = np.linspace(y_data.min(), y_data.max(), 50)
        X, Y = np.meshgrid(x_lin, y_lin)

        # Interpolate IVs
        Z = griddata(points=(x_data, y_data), values=z_data, xi=(X, Y), method="linear")
        # Z = np.nan_to_num(Z, nan=np.nanmean(z_data))

        # Create the surface plot
        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale=color_scale.lower(),
                    colorbar=dict(
                        title="IV",
                    ),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title=dict(text="Implied Volatility Surface", font=dict(size=20)),
            scene=dict(
                xaxis_title=dict(text=f"{index_col.capitalize()}", font=dict(size=14)),
                yaxis_title=dict(text="Time to Maturity (Years)", font=dict(size=14)),
                zaxis_title=dict(text="Implied Volatility", font=dict(size=14)),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectratio=dict(x=1, y=1, z=0.7),
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=30),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        with st.expander("Understanding the Surface Plot"):
            st.markdown("""
            The volatility surface shows how implied volatility varies across different strike prices and maturities.
            
            - **X-axis**: Shows the strike price or moneyness (strike/spot)
            - **Y-axis**: Shows time to maturity in years
            - **Z-axis**: Shows the implied volatility level
            - **Color**: Represents the magnitude of implied volatility
            
            The shape of the surface reveals important market dynamics:
            - **Volatility smile/skew**: Curvature across strikes
            - **Term structure**: How volatility changes with maturity
            """)

elif visualization_type == "Heatmap":
    # Create a pivot table for the heatmap
    pivot_data = data.pivot_table(
        values="impliedVolatility", index="strike", columns="T", aggfunc="mean"
    )

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=color_scale.lower(),
            colorbar=dict(title="IV"),
        )
    )

    fig.update_layout(
        title=dict(text=f"Implied Volatility Heatmap - {ticker}", font=dict(size=20)),
        xaxis_title=dict(text="Time to Maturity (Years)", font=dict(size=14)),
        yaxis_title=dict(text="Strike Price", font=dict(size=14)),
        width=800,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanation
    with st.expander("Understanding the Heatmap"):
        st.markdown("""
        The heatmap provides a 2D view of the volatility surface:
        
        - **X-axis**: Time to maturity in years
        - **Y-axis**: Strike price
        - **Color**: Intensity represents implied volatility level
        
        This visualization makes it easier to identify patterns across strikes and maturities.
        """)

elif visualization_type == "Smile Curves":
    # Get unique maturities and sort them
    maturities = sorted(data["T"].unique())

    # Create a figure with multiple lines (one for each maturity)
    fig = go.Figure()

    # Color scale for the lines
    colors = px.colors.sequential.Viridis
    color_step = max(1, len(colors) // len(maturities))

    # Add a line for each maturity
    for i, maturity in enumerate(maturities):
        maturity_data = data[data["T"] == maturity]
        # Sort by moneyness for smooth curves
        maturity_data = maturity_data.sort_values("moneyness")

        # Convert maturity to days for better readability
        days = int(maturity * 252)

        fig.add_trace(
            go.Scatter(
                x=maturity_data["moneyness"],
                y=maturity_data["impliedVolatility"],
                mode="lines+markers",
                name=f"{days} days",
                line=dict(width=3, color=colors[min(i * color_step, len(colors) - 1)]),
                marker=dict(size=6, opacity=0.7),
            )
        )

    # Add a vertical line at moneyness = 1 (at-the-money)
    fig.add_vline(
        x=1, line_dash="dash", line_color="rgba(0,0,0,0.5)", annotation_text="ATM"
    )

    fig.update_layout(
        title=dict(text=f"Volatility Smile Curves - {ticker}", font=dict(size=20)),
        xaxis=dict(
            title=dict(text="Moneyness (Strike/Spot)", font=dict(size=14)),
            tickformat=".2f",
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            title=dict(text="Implied Volatility", font=dict(size=14)),
            tickformat=".1%",
            gridcolor="rgba(0,0,0,0.1)",
        ),
        legend=dict(
            title=dict(text="Time to Maturity", font=dict(size=12)),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        width=800,
        height=500,
        plot_bgcolor="rgba(255,255,255,1)",
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanation
    with st.expander("Understanding Volatility Smiles"):
        st.markdown("""
        Volatility smiles show how implied volatility varies with moneyness for a specific maturity:
        
        - **X-axis**: Moneyness (Strike/Spot), where 1.0 is at-the-money
        - **Y-axis**: Implied volatility
        - **Each line**: Represents a different maturity
        
        Key patterns to look for:
        - **Smile**: Higher IV for both in-the-money and out-of-the-money options
        - **Skew**: Higher IV for out-of-the-money puts (moneyness < 1)
        - **Term structure**: How the smile/skew changes with maturity
        """)

elif visualization_type == "ATM Term Structure":
    # Approximate ATM: within 5% of spot
    spot_price = data["underlying_price"].iloc[0]
    atm_data = data[np.abs(data["strike"] - spot_price) / spot_price < 0.05]

    if atm_data.empty:
        st.warning("No at-the-money options found. Try adjusting your filters.")
    else:
        # group by maturity and take mean IV for ATM strikes and sort
        term_structure = atm_data.groupby("T")["impliedVolatility"].mean().reset_index()
        term_structure = term_structure.sort_values("T")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=term_structure["T"],
                y=term_structure["impliedVolatility"],
                mode="lines+markers",
                name="ATM IV",
                line=dict(color="#1E88E5", width=3),
                marker=dict(size=8, color="#1E88E5", line=dict(width=2, color="white")),
            )
        )

        fig.update_layout(
            title=dict(
                text=f"ATM Implied Volatility Term Structure - {ticker}",
                font=dict(size=20),
            ),
            xaxis=dict(
                title=dict(text="Time to Maturity (Years)", font=dict(size=14)),
                gridcolor="rgba(0,0,0,0.1)",
            ),
            yaxis=dict(
                title=dict(text="Implied Volatility", font=dict(size=14)),
                tickformat=".1%",
                gridcolor="rgba(0,0,0,0.1)",
            ),
            width=800,
            height=500,
            plot_bgcolor="rgba(255,255,255,1)",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Understanding the Term Structure"):
            st.markdown("""
            The term structure shows how implied volatility changes with time to maturity for at-the-money options:
            
            - **X-axis**: Time to maturity in days
            - **Y-axis**: Implied volatility for ATM options
            
            Common patterns:
            - **Upward sloping**: Longer-dated options have higher IV (normal market)
            - **Downward sloping**: Shorter-dated options have higher IV (market expecting near-term events)
            - **Humped**: Medium-term options have highest IV (mixed expectations)
            """)

with st.expander("Learn More About Volatility Analysis"):
    st.markdown("""
    ### Key Concepts in Volatility Analysis
    
    #### Implied Volatility
    Implied volatility is the market's forecast of a likely movement in the underlying asset's price. It's derived from option prices using the Black-Scholes model or similar pricing models.
    
    #### Volatility Surface
    The volatility surface is a three-dimensional representation of implied volatility across different strike prices and maturities. It reveals important market dynamics and expectations.
    
    #### Key Patterns
    
    1. **Volatility Smile/Skew**:
        - **Smile**: Higher IV for both in-the-money and out-of-the-money options
        - **Skew**: Higher IV for out-of-the-money puts, reflecting crash risk
    
    2. **Term Structure**:
        - **Normal (investment skew)**: Upward sloping (longer-dated options have higher IV)
        - **Inverted**: Downward sloping (shorter-dated options have higher IV)
        - **Humped**: Medium-term options have highest IV
    
    3. **Surface Dynamics**:
        - **Parallel shifts**: Overall IV level changes
        - **Twist**: Term structure changes
        - **Butterfly**: Smile/skew changes
    
    #### Trading Applications
    
    - **Relative value**: Identify mispriced options
    - **Volatility arbitrage**: Trade differences in IV across strikes/maturities
    - **Risk management**: Understand exposure to volatility changes
    - **Strategy selection**: Choose appropriate strategies based on the volatility environment
    """)

# Footer
st.markdown("---")
st.caption(
    "Data provided by Yahoo Finance. Calculations based on the Black-Scholes model."
)
