import numpy as np
from scipy.stats import norm
from tools.option import Option
from abc import ABC
from enum import Enum


# All calculations are based on the Black-Scholes in Options and Volatility by Sheldon Natenberg pg 350
class OptionSettlementType(Enum):
    STOCK = "stock"
    FUTURE_FS = "future_fs"  # futures with futures settlement
    FUTURE_CS = "future_cs"  # futures with cash settlement
    FX = "fx"


class BlackScholesOption(Option, ABC):
    def __init__(
        self,
        S,
        K,
        T,
        r,
        sigma,
        option_settlement_type: OptionSettlementType = OptionSettlementType.STOCK,
        rf=None,
        q=0,
    ):
        """
        S is the stock price
        K is the strike price
        T is the time to expiration in years
        r is the risk-free rate
        sigma is the volatility
        option_settlement_type is the type of option settlement
        rf is the foreign risk-free rate
        q is the dividend yield
        """
        super().__init__(S, K, T, r, sigma)
        self.S = S
        self.K = K
        self.T = T  # in years
        self.r = r
        self.sigma = sigma
        self.option_settlement_type = option_settlement_type

        match self.option_settlement_type:
            case OptionSettlementType.FUTURE_FS:
                self.b = self.r = 0
            case OptionSettlementType.FUTURE_CS:
                self.b = 0
            case OptionSettlementType.FX:
                if rf is None:
                    raise ValueError("rf must be provided for FX options")
                self.b = self.r - rf
            case _:
                self.b = self.r

        self.b = self.b - q  # approximate discount of expected dividends

    def delta(self):
        pass

    def theta(self):
        pass

    def rho(self):
        pass

    def charm(self):
        pass

    def _d1(self):
        return (np.log(self.S / self.K) + (self.b + self.sigma**2 / 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def gamma(self):
        return (
            np.exp((self.b - self.r) * self.T)
            * norm.pdf(self._d1())
            / (self.S * self.sigma * np.sqrt(self.T))
        )

    def vega(self):
        return (
            self.S
            * np.exp((self.b - self.r) * self.T)
            * norm.pdf(self._d1())
            * np.sqrt(self.T)
        ) / 100

    def vanna(self):
        """
        Vanna is the derivative of delta with respect to the volatility.
        """
        return (
            -np.exp((self.b - self.r) * self.T)
            * norm.pdf(self._d1())
            * self._d2()
            / self.S
            * self.sigma
        )

    def volga(self):
        """
        Volga (vomma) is the derivative of vega with respect to the volatility.
        """
        return self.vega() * self._d1() * self._d2() / self.sigma

    def vomma(self):
        """
        see volga
        """
        return self.volga()

    def speed(self):
        """
        Speed is the derivative of gamma with respect to the stock price.
        """
        return (
            -self.gamma() / self.S * (1 + (self._d1() / self.sigma * np.sqrt(self.T)))
        )

    def color(self):
        """
        Color is the derivative of gamma with respect to the passage of time.
        """
        return self.gamma() * (
            self.r
            - self.b
            + (
                self.b * self._d1() / self.sigma * np.sqrt(self.T)
                + ((1 - (self._d1() * self._d2())) / (2 * self.T))
            )
        )

    def zomma(self):
        """
        Zomma is the derivative of gamma with respect to the volatility.
        """
        return self.gamma() * ((self._d1() * self._d2() - 1) / self.sigma)

    def delta_50(self):
        """
        Price at which delta will be 50%
        """
        return self.K * np.exp(-self.b * self.T - self.sigma**2 * self.T / 2)

    def max_gamma(self):
        """
        Price at which gamma will be maximum
        """
        return self.K * np.exp(-self.b * self.T - 3 * self.sigma**2 * self.T / 2)

    def max_theta(self):
        """
        Price at which theta will be maximum
        """
        return self.K * np.exp(self.b * self.T + self.sigma**2 * self.T / 2)

    def max_vega(self):
        """
        Price at which vega will be maximum
        """
        return self.K * np.exp(-self.b * self.T + self.sigma**2 * self.T / 2)

    def calculate_price(self):
        # For Call option
        if self.T <= 0:
            # Handle expired options
            return (
                max(0, self.S - self.K)
                if isinstance(self, Call)
                else max(0, self.K - self.S)
            )

        # For very small volatility
        if self.sigma < 1e-10:
            if isinstance(self, Call):
                return max(
                    0,
                    self.S * np.exp((self.b - self.r) * self.T)
                    - self.K * np.exp(-self.r * self.T),
                )
            else:
                return max(
                    0,
                    self.K * np.exp(-self.r * self.T)
                    - self.S * np.exp((self.b - self.r) * self.T),
                )

        # Normal case
        d1 = self._d1()
        d2 = self._d2()

        if isinstance(self, Call):
            return self.S * np.exp((self.b - self.r) * self.T) * norm.cdf(
                d1
            ) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(
                (self.b - self.r) * self.T
            ) * norm.cdf(-d1)


class Call(BlackScholesOption):
    def __init__(self, S, K, T, r, sigma, q=0):
        super().__init__(S, K, T, r, sigma, q=q)

    def delta(self):
        return norm.cdf(self._d1()) * np.exp((self.b - self.r) * self.T)

    def theta(self):
        theta_value = (
            -(
                self.S
                * np.exp((self.b - self.r) * self.T)
                * norm.pdf(self._d1())
                * self.sigma
            )
            / (2 * np.sqrt(self.T))
            - (self.b - self.r)
            * self.S
            * np.exp((self.b - self.r) * self.T)
            * norm.cdf(self._d1())
            - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2())
        )
        return theta_value / 365

    def rho(self):
        val = (
            (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self._d2()))
            if self.b != 0
            else -self.calculate_price() * self.T
        )
        return val / 100

    def charm(self):
        """
        Charm is the derivative of delta with respect to the passage of time.
        """
        return -np.exp((self.b - self.r) * self.T) * (
            norm.pdf(self._d1())
            * (
                self.b / (self.sigma * np.sqrt(self.T))
                - self._d2() / (2 * self.T)
                + (self.b - self.r) * norm.cdf(self._d1())
            )
        )


class Put(BlackScholesOption):
    def __init__(self, S, K, T, r, sigma, q=0):
        super().__init__(S, K, T, r, sigma, q=q)

    def delta(self):
        return (norm.cdf(self._d1()) - 1) * np.exp((self.b - self.r) * self.T)

    def theta(self):
        theta_value = (
            -(
                self.S
                * np.exp((self.b - self.r) * self.T)
                * norm.pdf(self._d1())
                * self.sigma
            )
            / (2 * np.sqrt(self.T))
            + (self.b - self.r)
            * self.S
            * np.exp((self.b - self.r) * self.T)
            * norm.cdf(self._d1())
            + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2())
        )
        return theta_value / 365

    def rho(self):
        val = (
            -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self._d2()))
            if self.b != 0
            else -self.calculate_price() * self.T
        )
        return val / 100

    def charm(self):
        """
        Charm is the derivative of delta with respect to the passage of time.
        """
        return -np.exp((self.b - self.r) * self.T) * (
            norm.pdf(self._d1())
            * (
                self.b / (self.sigma * np.sqrt(self.T))
                - self._d2() / (2 * self.T)
                - (self.b - self.r) * norm.cdf(self._d1())
            )
        )
