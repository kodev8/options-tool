# Options Analytics Suite

A comprehensive options pricing and analysis tool built with Python and Streamlit, providing traders, analysts, and students with powerful tools to understand options pricing, Greeks, and volatility dynamics.
Try it out [here](https://options-tool.streamlit.app/).

![Options Tool](https://img.shields.io/badge/Options-Tool-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![Yahoo Finance](https://img.shields.io/badge/Yahoo-Finance-yellow)

## Project Structure
```
options-tool/
├── main.py                  # Main application entry point
├── pages/                   # Streamlit pages
│   ├── main_page.py         # Home page
│   ├── bs_page.py           # Black-Scholes calculator
│   ├── vol_surface_page.py  # Volatility surface visualization
│   └── styles/              # CSS styles
├── tools/                   # Core functionality
│   ├── bs_calc.py           # Black-Scholes implementation
│   ├── option.py            # Option base class
│   └── volatility_calc.py   # Volatility calculations
└── README.md                # This file
```

## Features

### Black-Scholes Calculator
- Price European call and put options with the Black-Scholes model
- Calculate all Greeks (Delta, Gamma, Theta, Vega, Rho)
- Visualize second-order Greeks (Charm, Vanna, Volga, etc.)
- Support for different underlying assets (stocks, futures, FX)
- Dividend yield adjustments

### Volatility Surface Analysis
- Interactive 3D visualization of implied volatility surfaces
- Multiple visualization types:
  - Surface plots (by strike or moneyness)
  - Heatmaps for easier pattern recognition
  - Volatility smile curves by maturity
  - ATM term structure analysis
- Filtering by strike range, moneyness, and option type
- Real-time data from Yahoo Finance

### Option Chain Analysis
- Explore full option chains for any ticker
- Calculate implied volatility for all strikes and expirations
- Identify mispriced options
- Analyze bid-ask spreads and open interest

## Installation
Clone the repository
```bash
git clone https://github.com/kodev8/options-tool.git
cd options-tool
```
Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
Run the application
streamlit run main.py
```

## Usage

### Black-Scholes Calculator

The calculator allows you to:
- Enter stock price, strike price, time to expiration, risk-free rate, and volatility
- View option prices and all Greeks in real-time
- Perform sensitivity analysis to see how option prices change with different parameters
- Calculate advanced Greeks for sophisticated risk management

### Volatility Surface

The volatility surface tool provides:
- Interactive 3D visualization of implied volatility across strikes and maturities
- Multiple visualization options to analyze volatility patterns
- Filtering capabilities to focus on specific areas of interest
- Educational content to help understand volatility dynamics

### Data Sources

The application uses:
- Yahoo Finance API for real-time market data
- Black-Scholes model for theoretical pricing
- Numerical optimization for implied volatility calculation

## Technical Details

### Models Implemented

- **Black-Scholes-Merton Model**: The core pricing model with extensions for:
  - Dividend-paying stocks
  - Futures options
  - FX options

  <i>**Note**: 
  - The tool currently only supports European options. American options will be supported in future updates.
  - The tool currently only supports stock options. Future updates will include options on futures and other assets.</i>

### Greeks Calculation

The tool calculates:
- **First-order Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Second-order Greeks**: Charm, Vanna, Volga/Vomma, Speed, Color, Zomma

### Implied Volatility Calculation

Implied volatility is calculated using:
- Optimization-based approach for robust convergence
- Fallback to grid search for edge cases
- Proper handling of numerical issues

