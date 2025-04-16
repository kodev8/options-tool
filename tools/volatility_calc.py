import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from datetime import datetime, timedelta
from tools.bs_calc import Call, Put
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize_scalar



@st.cache_data
def get_ticker_data(ticker):
    data = yf.Ticker(ticker)
    return data


@st.cache_data
def get_data(
    ticker, 
    start_date: datetime.date = datetime.date(datetime.today() + timedelta(days=5)), 
    end_date: datetime.date = None
    ) -> pd.DataFrame | None:
    # Todo convert to pickled data
    data = yf.Ticker(ticker)

    # yf quotes dividend yield as a percentage 
    try:

        trial_dividend_keys = ["dividendYield", "fiveYearAvgDividendYield", "trailingAnnualDividendYield"]
        for key in trial_dividend_keys:
            dividend_yield = data.info.get(key)

            if dividend_yield is not None:
                if dividend_yield > 1:
                    dividend_yield = dividend_yield / 100
                break

    except:
        dividend_yield = 0


    if end_date <= start_date:
        st.error("End date must be greater than start date")
        return None

    end_date = end_date if end_date else start_date + timedelta(days=30)
    filtered = [
        x
        for x in data.options
        if start_date <= datetime.date(datetime.strptime(x, "%Y-%m-%d")) <= end_date
    ]


    if len(filtered) == 0:
        st.error("No options found for the given ticker and date range")
        return None

    dfs = []
    current_underlying_price = data.history(period="5d")["Close"]
    if current_underlying_price.empty:
        st.error("No data found for the given ticker")
        return None

    current_underlying_price = current_underlying_price.iloc[-1]
    if current_underlying_price == 0:
        st.error("Current underlying price is 0")
        return None

    for f in filtered:
        calls = data.option_chain(f).calls
        calls["option_type"] = "C"
        puts = data.option_chain(f).puts
        puts["option_type"] = "P"

        df = pd.concat([calls, puts])
        df["exp_date"] = datetime.strptime(f, "%Y-%m-%d")
        df["T"] = (df["exp_date"] - datetime.today()).dt.days / 252

        # moneyness is the strike price divided by the current price
        # TODO: use atm strike or underlying price? should be close enough
        df["underlying_price"] = current_underlying_price
        df["moneyness"] = df["strike"] / df["underlying_price"]
        df["mid_price"] = (df["bid"] + df["ask"]) / 2

        # filter out wide bid-ask spreads and no open interest
        # df = df[(df["ask"] - df["bid"]) / df["mid_price"] < 0.3]
        # df = df[df["openInterest"] > 0]

        df = df[df['mid_price'] > 0]
        df = df[(df['moneyness'] > 0.7) & (df['moneyness'] < 1.3)]

        df["impliedVolatility"] = df.apply(
            lambda row: implied_volatility(
                row["mid_price"],
                row["underlying_price"],
                row["strike"],
                row["T"],
                option_type=row["option_type"],
                dividend_yield=dividend_yield,
            ),
            axis=1,
        )

        df = df[
            [
                "contractSymbol",
                "strike",
                "impliedVolatility",
                "option_type",
                "exp_date",
                "T",
                "underlying_price",
                "moneyness",
                "mid_price",
            ]
        ]
    
        dfs.append(df)

    if not dfs:
        return None

    final = pd.concat(dfs)
    # removes extreme IV values
    final = final[
        (final["impliedVolatility"] > 0.01) & (final["impliedVolatility"] < 10.0)
    ]

    return final


@st.cache_data
def get_risk_free_rate():
    try:
        risk_free_rate = yf.Ticker("^TNX").info["regularMarketPreviousClose"] / 100
    except:
        risk_free_rate = 0.05  # default if unable to fetch
    return round(risk_free_rate, 2)


def implied_volatility_brentq(
    option_price,
    underlying_price,
    strike,
    maturity,
    option_type="C",
    r=None,
    dividend_yield=0,
):
    if not r:
        r = get_risk_free_rate()

    if option_price <= 0 or maturity <= 0:
        return np.nan

    low = 1e-6
    high = 10

    def objective(sigma):
        if option_type == "C":
            call = Call(underlying_price, strike, maturity, r, sigma, q=dividend_yield)
            return call.calculate_price() - option_price
        else:
            put = Put(underlying_price, strike, maturity, r, sigma, q=dividend_yield)
            return put.calculate_price() - option_price

    try:
        # no sign change i.e. we can't guarantee root in interval to avoid value error
        return brentq(objective, low, high)
    except:
        return np.nan


def implied_volatility(
    option_price,
    underlying_price,
    strike,
    maturity,
    option_type="C",
    r=None,
    dividend_yield=0,
):
    if not r:
        r = get_risk_free_rate()

    if option_price <= 0 or maturity <= 0:
        return np.nan

    # Use optimization approach for better convergence
    
    def objective(sigma):
        try:
            if option_type == "C":
                price = Call(underlying_price, strike, maturity, r, sigma, q=dividend_yield).calculate_price()
            else:
                price = Put(underlying_price, strike, maturity, r, sigma, q=dividend_yield).calculate_price()
            return abs(price - option_price)
        except:
            return float('inf')
    
    try:
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        if result.success:
            return result.x
        else:
            return np.nan
    except:
        # If optimization fails, fall back to a grid search
        best_sigma = np.nan
        min_diff = float('inf')
        
        for sigma in np.linspace(0.01, 2.0, 100):
            try:
                if option_type == "C":
                    price = Call(underlying_price, strike, maturity, r, sigma, q=dividend_yield).calculate_price()
                else:
                    price = Put(underlying_price, strike, maturity, r, sigma, q=dividend_yield).calculate_price()
                
                diff = abs(price - option_price)
                if diff < min_diff:
                    min_diff = diff
                    best_sigma = sigma
            except:
                continue
        
        return best_sigma