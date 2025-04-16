from abc import ABC, abstractmethod


class Option(ABC):
    def __init__(self, ticker, strike, maturity, r, sigma):
        self.ticker = ticker
        self.strike = strike
        self.maturity = maturity
        self.r = r
        self.sigma = sigma

    @abstractmethod
    def calculate_price(self):
        pass

    @abstractmethod
    def delta(self):
        pass

    @abstractmethod
    def gamma(self):
        pass

    @abstractmethod
    def theta(self):
        pass

    @abstractmethod
    def vega(self):
        pass

    @abstractmethod
    def rho(self):
        pass
