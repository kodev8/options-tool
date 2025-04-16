import pandas as pd
import numpy as np
from tools.bs_calc import Call, Put
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tools.volatility_calc import get_data, implied_volatility, get_risk_free_rate
from datetime import datetime, timedelta
from streamlit.components.v1 import html

st.set_page_config(layout="wide", page_title="Black-Scholes Calculator", page_icon="🔢")


with open("pages/styles/custom.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.markdown(
    '<div class="main-header">Black-Scholes Calculator</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="info-text">Calculate the price of a call or put option using the Black-Scholes model</div>',
    unsafe_allow_html=True,
)

default_values = {
    "S": 100.0,
    "K": 100.0,
    "T": 1.0,
    "r": get_risk_free_rate(),
    "q": 0.0,
    "sigma": 0.2,
}

greeks_symbold = {
    "Delta": "Δ",
    "Gamma": "Γ",
    "Theta": "Θ",
    "Vega": "V",
    "Rho": "ρ",
}

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Option Prices & Greeks",
        "Sensitivity Analysis",
        "Advanced Greeks",
        "PnL Heatmap",
        "Compute Option Chain ",
    ]
)

# Sidebar for inputs
with st.sidebar:
    st.write("## Inputs")

    S = st.number_input("Stock Price", value=default_values["S"], min_value=0.1)
    K = st.number_input("Strike Price", value=default_values["K"], min_value=0.1)

    time_to_expiration_cols = st.columns(2)
    with time_to_expiration_cols[0]:
        T = st.number_input(
            "Time to Expiration", value=default_values["T"], min_value=0.1
        )
    with time_to_expiration_cols[1]:
        time_format = st.selectbox("Time Format", ["Years", "Months", "Weeks", "Days"])
        if time_format == "Days":
            T = T / 365
        elif time_format == "Weeks":
            T = (T * 7) / 365
        elif time_format == "Months":
            T = (T * 30) / 365
        elif time_format == "Years":
            T = T
        else:
            raise ValueError("Invalid time format")

    q = st.number_input(
        "Dividend Yield", value=default_values["q"], min_value=0.0, max_value=1.0
    )
    r = st.number_input(
        "Risk-Free Rate", value=default_values["r"], min_value=0.0, max_value=1.0,
    )
    sigma = st.slider(
        "Volatility",
        min_value=0.01,
        max_value=2.0,
        value=default_values["sigma"],
        step=0.01,
    )

    year_time = T

    call = Call(S, K, year_time, r, sigma)
    put = Put(S, K, year_time, r, sigma)

# Tab 1: Option Prices & Greeks
with tab1:
    # Price cards at the top
    # selected_data_cols = st.columns(5)
    # with selected_data_cols[0]:
    #     st.metric("Stock Price", f"${round(S, 4)}")
    # with selected_data_cols[1]:
    #     st.metric("Strike Price", f"${round(K, 4)}")
    # with selected_data_cols[2]:
    #     st.metric("Time to Expiration", f"{round(T, 4)} days")
    # with selected_data_cols[3]:
    #     st.metric("Risk-Free Rate", f"{round(r, 4)}")
    # with selected_data_cols[4]:
    #     st.metric("Volatility", f"{round(sigma, 4)}")

   

    price_cols = st.columns(2)
    with price_cols[0]:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card green font-bold">
                    <div class="">Call Price</div>
                    <div class="metric-value">${round(call.calculate_price(), 4)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with price_cols[1]:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card red font-bold">
                    <div class="">Put Price</div>
                    <div class="metric-value">${round(put.calculate_price(), 4)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("Option Parameters"):
        st.markdown(
            f"""
            <div class="highlight metric-container">
            <div class="metric-card">
                <div class="metric-label">Stock Price</div>
                <div class="metric-value">${round(S, 4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Strike Price</div>
                <div class="metric-value">${round(K, 4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Time to Expiration</div>
                <div class="metric-value">{round(T, 4)} years</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk-Free Rate</div>
                <div class="metric-value">{round(r, 4)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{round(sigma, 4)}</div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    with st.expander("Put-Call Parity"):
        st.write(
            "The put-call parity is a relationship between the price of a call option and the price of a put option, both with the same underlying asset, strike price, and expiration date."
        )
        st.latex(r"C + Ke^{-rt}= P + S")
        st.latex(
            r"C + K * e^{-rt} = "
            + str(round(call.calculate_price(), 4))
            + " + "
            + str(round(K * np.exp(-r * year_time), 4))
            + " = "
            + str(round(call.calculate_price() + K * np.exp(-r * year_time), 4))
        )
        st.latex(
            r"P + S = "
            + str(round(put.calculate_price(), 4))
            + " + "
            + str(round(S, 4))
            + " = "
            + str(round(put.calculate_price() + S, 4))
        )


    # Greeks table
    st.subheader("Option Greeks")

    # Create a combined DataFrame for both call and put
    greeks_df = pd.DataFrame(
        {
            f"Delta ({greeks_symbold['Delta']})": [call.delta(), put.delta(), f" ∑{greeks_symbold['Delta']} = {abs(call.delta() - put.delta())}"],
            f"Gamma ({greeks_symbold['Gamma']})": [call.gamma(), put.gamma(), f"C == {greeks_symbold['Gamma']}P"],
            f"Theta ({greeks_symbold['Theta']})": [call.theta(), put.theta(), f"{greeks_symbold['Gamma']} * {greeks_symbold['Theta']} = -ve"],
            f"Vega ({greeks_symbold['Vega']})": [call.vega(), put.vega(), f"{greeks_symbold['Vega']}C == {greeks_symbold['Vega']}P"],
            f"Rho ({greeks_symbold['Rho']})": [call.rho(), put.rho(), ""],
        },
        index=["Call", "Put", "Notes"],
    )


    greeks_df_display = greeks_df.copy()
    for col in greeks_df.columns:
        greeks_df_display.loc[["Call", "Put"], col] = greeks_df_display.loc[["Call", "Put"], col].apply(lambda x: f"{x:.4f}")
        
    greeks_df_display = greeks_df_display.astype(str)

    st.dataframe(greeks_df_display)

# Tab 2: Sensitivity Analysis
with tab2:
    # Allow user to select which parameter to analyze
    analysis_param = st.selectbox(
        "Select parameter to analyze",
        [
            "Stock Price",
            "Strike Price",
            "Time to Expiration",
            "Risk-Free Rate",
            "Volatility",
        ],
    )

    # Generate range of values based on selected parameter

    match analysis_param:
        case "Stock Price":
            current_value = S
            param_range = np.linspace(max(0.1, S * 0.5), S * 1.5, 100)
            call_prices = [
                Call(p, K, year_time, r, sigma).calculate_price() for p in param_range
            ]
            put_prices = [
                Put(p, K, year_time, r, sigma).calculate_price() for p in param_range
            ]
            x_label = "Stock Price"
        case "Strike Price":
            current_value = K
            param_range = np.linspace(max(0.1, K * 0.5), K * 1.5, 100)
            call_prices = [
            Call(S, p, year_time, r, sigma).calculate_price() for p in param_range
            ]
            put_prices = [
                Put(S, p, year_time, r, sigma).calculate_price() for p in param_range
            ]
            x_label = "Strike Price"
        case "Time to Expiration":
            current_value = T
            param_range = np.linspace(max(1, T * 0.5), T * 1.5, 100)
            call_prices = [
                Call(S, K, (p / 365) * 252, r, sigma).calculate_price() for p in param_range
            ]
            put_prices = [
                Put(S, K, (p / 365) * 252, r, sigma).calculate_price() for p in param_range
            ]
            x_label = "Time to Expiration (days)"
        case "Risk-Free Rate":
            current_value = r
            param_range = np.linspace(0, min(0.2, r * 2), 100)
            call_prices = [
                Call(S, K, year_time, p, sigma).calculate_price() for p in param_range
            ]
            put_prices = [
                Put(S, K, year_time, p, sigma).calculate_price() for p in param_range
            ]
            x_label = "Risk-Free Rate"

        case "Volatility":
            current_value = sigma
            param_range = np.linspace(0.01, min(2.0, sigma * 2), 100)
            call_prices = [Call(S, K, year_time, r, p).calculate_price() for p in param_range]
            put_prices = [Put(S, K, year_time, r, p).calculate_price() for p in param_range]
            x_label = "Volatility"

        case _:
            st.error("Invalid parameter")
            st.stop()


    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=call_prices,
            mode="lines",
            name="Call Price",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=put_prices,
            mode="lines",
            name="Put Price",
            line=dict(color="red"),
        )
    )

    fig.add_vline(
        x=current_value,
        line_dash="dash",
        line_color="gray",
        annotation_text="Current Value",
    )

    fig.update_layout(
        title=f"Option Price Sensitivity to {analysis_param}",
        xaxis_title=x_label,
        yaxis_title="Option Price",
        legend_title="Option Type",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Advanced Greeks
with tab3:

    adv_cols = st.columns(2)
    with adv_cols[0]:
        st.markdown("### Maximum Values", help="price at which the greek is at its maximum")
        max_df = pd.DataFrame(
            {
                "Max Gamma": [round(call.max_gamma(), 4)],
                "Max Theta": [round(call.max_theta(), 4)],
                "Max Vega": [round(call.max_vega(), 4)],
            },
            index=["Value"],
        )
        st.dataframe(max_df)

    with adv_cols[1]:
        st.write("### Higher Order Derivatives")
        higher_df = pd.DataFrame(
            {
                "Vanna": [round(call.vanna(), 4)],
                "Volga": [round(call.volga(), 4)],
                "Speed": [round(call.speed(), 4)],
                "Color": [round(call.color(), 4)],
                "Zomma": [round(call.zomma(), 4)],
            },
            index=["Value"],
        )
        st.dataframe(higher_df)

    # Add explanations
    with st.expander("Greek Definitions"):
        st.markdown("""
        - **Vanna**: Measures the rate of change of delta with respect to volatility
        - **Volga**: Measures the rate of change of vega with respect to volatility
        - **Speed**: Measures the rate of change of gamma with respect to the underlying price
        - **Color**: Measures the rate of change of gamma with respect to time
        - **Zomma**: Measures the rate of change of gamma with respect to volatility
        """)

# Tab 4: PnL Heatmap
with tab4:
    st.subheader("PnL Heatmap")

    with st.sidebar:
        st.divider()
        st.write("### PnL Heatmap")
        heatmap_price_cols = st.columns(2)
        with heatmap_price_cols[0]:
            min_spot_price = st.number_input("Min Spot Price", value=0.75 * K, min_value=0.01, step=0.01)
        with heatmap_price_cols[1]:
            max_spot_price = st.number_input("Max Spot Price", value=1.25 * K, min_value=0.01, step=0.01)

        heatmap_vol_slider = st.slider(
            "Volatility Filter",
            min_value=0.01,
            max_value=2.0,
            value=(0.01, 2.0),
        )

    GRID_SIZE = 12

    spot_price_vals = np.linspace(min_spot_price, max_spot_price, GRID_SIZE)
    vol_vals = np.linspace(heatmap_vol_slider[0], heatmap_vol_slider[1], GRID_SIZE)
    
    call_pills = st.pills(label="Call Heatmap options", options=["Show PnL", "Show Price"], default="Show PnL")
    
    # https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-numpy#:~:text=The%20purpose%20of%20meshgrid%20is,the%20x%20and%20y%20directions.
    # create a meshgrid of strike and vol
    spot_price_grid, vol_grid = np.meshgrid(spot_price_vals, vol_vals)
    call_prices = np.zeros((len(spot_price_grid), len(vol_grid)))
    put_prices = np.zeros((len(spot_price_grid), len(vol_grid)))

    for i in range(len(spot_price_grid)):
        for j in range(len(vol_grid)):
            call_prices[i, j] = Call(spot_price_grid[i, j], K, T / 252, r, vol_grid[i, j]).calculate_price()
            put_prices[i, j] = Put(spot_price_grid[i, j], K, T / 252, r, vol_grid[i, j]).calculate_price()

    # CALL HEATMAP
    call_purchase_price = st.number_input("Call Purchase Price", value=0.0, min_value=0.0, step=0.01)
    fig_call = go.Figure()
    # ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
    fig_call.add_trace(go.Heatmap(x=spot_price_vals, y=vol_vals, z=call_prices, colorscale="Cividis", colorbar=dict(title="Call Price")))
    fig_call.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        title="Call PnL Heatmap",
        height=500,
        # x_axis=dict(tickmode="linear", dtick=10),
        # y_axis=dict
        # (tickmode="linear", dtick=0.1),
    )

    for i in range(len(spot_price_vals)):
        for j in range(len(vol_vals)):
            # call_prices[i, j] = call_prices[i, j] - call_purchase_price
            fig_call.add_annotation(
                x=spot_price_vals[i],
                y=vol_vals[j],
                text=f"{call_prices[i, j]:.2f}" if call_pills == "Show Price" else f"{call_purchase_price - call_prices[i, j]:.2f}",
                showarrow=False,
                font=dict(
                    color="white" if call_purchase_price == 0 else "green" if call_prices[i, j] < call_purchase_price else "red",
                    size=12
                )
            )
    st.plotly_chart(fig_call, use_container_width=True)
    
    st.divider()

    # PUT HEATMAP
    # plot 2d grid of put prices
    put_pills = st.pills(label="Put Heatmap options", options=["Show PnL", "Show Price"], default="Show PnL", key="put_pills")
    put_purchase_price = st.number_input("Put Purchase Price", value=0.0, min_value=0.0, step=0.01)
    fig_put = go.Figure()
    fig_put.add_trace(go.Heatmap(x=spot_price_vals, y=vol_vals, z=put_prices, colorscale="Cividis", colorbar=dict(title="Put Price")))
    fig_put.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        title="Put PnL Heatmap",
        # x_axis=dict(tickmode="linear", dtick=10),
        # y_axis=dict(tickmode="linear", dtick=0.1),

        )
    
    for i in range(len(spot_price_vals)):
        for j in range(len(vol_vals)):
            fig_put.add_annotation(
                x=spot_price_vals[i],
                y=vol_vals[j],
                text=f"{put_prices[i, j]:.2f}" if put_pills == "Show Price" else f"{put_purchase_price - put_prices[i, j]:.2f}",
                showarrow=False,
                font=dict(
                    color="white" if put_purchase_price == 0 else "green" if put_prices[i, j] < put_purchase_price else "red",
                    size=12
                )
            )

    st.plotly_chart(fig_put, use_container_width=True)



# Tab 5: Option Chains
with tab5:
    st.subheader("Option Chains")
    st.write("Option chains for a given ticker with greeks and IV computed using the Black-Scholes model.")
    ticker = st.text_input("Ticker", value="AAPL")
    data = get_data(
        ticker, end_date=datetime.date(datetime.now() + timedelta(days=1080))
    )

    underlying_price = data["underlying_price"].iloc[0]

    st.metric("Underlying Price", round(underlying_price, 2))
    if data is None:
        st.error("No data found for the given ticker.")
        st.stop()


    def add_greeks(row, option_type):
        if option_type == "C":
            option = Call(
                row["underlying_price"],
                row["strike"],
                row["T"],
                get_risk_free_rate(),
                row["impliedVolatility"],
            )
        else:
            option = Put(
                row["underlying_price"],
                row["strike"],
                row["T"],
                get_risk_free_rate(),
                row["impliedVolatility"],
            )

        row["delta"] = option.delta()
        row["gamma"] = option.gamma()
        row["theta"] = option.theta()
        row["vega"] = option.vega()
        row["rho"] = option.rho()
        return row

    tabs_strs = [x.strftime("%Y-%m-%d") for x in data["exp_date"].unique()]
    tabs = st.tabs(tabs_strs)
    for tab in range(len(tabs)):
        with tabs[tab]:
            data_ = data[data["exp_date"] == tabs_strs[tab]]
            data_ = data_[[col for col in data.columns if col not in ("exp_date")]]
            puts = data_[data_["option_type"] == "P"]
            calls = data_[data_["option_type"] == "C"]


            calls = calls.apply(lambda row: add_greeks(row, "C"), axis=1)
            puts = puts.apply(lambda row: add_greeks(row, "P"), axis=1)

            # remove option_type
            calls = calls.drop(columns=["option_type", "underlying_price"])
            puts = puts.drop(columns=["option_type", "underlying_price"])

            calls.set_index(["strike"], inplace=True)
            puts.set_index(["strike"], inplace=True)

            merged = calls.join(
                puts, on="strike", how="outer", lsuffix="_call", rsuffix="_put"
            )
            merged = merged.reset_index()

            # Reorder columns: calls on left, then strike, then puts on right
            call_cols = [col for col in merged.columns if col.endswith("_call")]
            put_cols = [col for col in merged.columns if col.endswith("_put")]

            cols = call_cols + ["strike"] + put_cols
            ordered = merged[cols]
            ordered = ordered.sort_values(by="strike")

            # create multi index for call and put columns
            call_tuples = [("Call", col.split("_call")[0]) for col in call_cols]
            put_tuples = [("Put", col.split("_put")[0]) for col in put_cols]
            strike_tuple = [("", "strike")]  # Empty string for top level to keep strike separate

            all_tuples = call_tuples + strike_tuple + put_tuples
            multi_index = pd.MultiIndex.from_tuples(all_tuples)
            ordered.columns = multi_index


            strike_col = [col for col in ordered.columns if col[1] == "strike"]
            other_cols = [col for col in ordered.columns if col not in strike_col]

            format_dict = {col: "{:.1f}" for col in strike_col}
            # format_dict.update({col: "{:.4f}" for col in other_cols})
            strike_col_idx = [i for i, col in enumerate(ordered.columns) if col[1] == "strike"]

            def style_strike(df):
                styles = []
                for _ in range(len(df)):
                    row_styles = []
                    for i in range(len(df.columns)):
                        if i in strike_col_idx:
                            row_styles.append("background-color: #f0f0f0; text-align: center")
                        else:
                            row_styles.append("text-align: center")
                    styles.append(row_styles)
                return pd.DataFrame(styles, columns=df.columns, index=df.index)

            styled_df = (
                ordered.style
                .format(format_dict)
                .set_table_styles(
                    [{'selector': 'th', 'props': [('text-align', 'center')]}]
                )
                .set_properties(**{'text-align': 'center'})  # center all cells
                .apply(style_strike, axis=None)
            )
            

            st.dataframe(styled_df)


html("""
<script defer type="text/javascript">
    const doc = window.parent.document
     let PnlTab = null
     let firstRun = true
    

    const toggleShow = (show) =>  {
            x = doc.evaluate("//h3[contains(., 'PnL')]", doc, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null ).singleNodeValue
            curr_parent = x.parentElement.parentElement.parentElement
            s1 = curr_parent.nextElementSibling
            s2 = s1?.nextElementSibling
         
            curr_parent.style.opacity = show ? 1 : 0
            s1.style.opacity = show ? 1 : 0
            s2.style.opacity = show ? 1 : 0
    }
     

     doc.querySelectorAll("[role='tab']").forEach(r => {
            if (r.innerText.includes('PnL')){
                PnlTab = r
                PnlTab.addEventListener('click', () => toggleShow(true))
                if (PnlTab.getAttribute('aria-selected') === 'true'){
                    toggleShow(true)
                 } else{
                  toggleShow(false)
                }
            }else{
                r.addEventListener('click', () => {
                    toggleShow(false)
               })
            }
         
     })
           
</script>
""",)
