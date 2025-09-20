import numpy as np
import pandas as pd

def credit_fv_delta(principal, maturity, credit_spread, interest_rates):
    """
    Compute the change in fair value (delta) for a single credit
    when the benchmark interest rate shifts.

    Parameters
    ----------
    principal : float
        Loan principal amount.
    maturity : int
        Total number of installments from t=0.
    credit_spread : float
        Loan-specific spread over the benchmark (annual).
    interest_rates : (float, float)
        Benchmark rates (annual) at t=0 and t=1.

    Returns
    -------
    float : delta value (fair value at t=1 under rate_1 - fair value at t=1 under rate_0)
    """
    # Effective discount rates (annual)
    rate0 = credit_spread + interest_rates[0]
    rate1 = credit_spread + interest_rates[1]

    # Monthly payment based on rate0 (coupon)
    r_month = rate0 / 12
    payment = principal * r_month / (1 - (1 + r_month) ** (-maturity))

    # Remaining payments at t=1
    R = maturity - 1

    # Present value at t=1 using rate0 and rate1 directly as per-period rates
    def pv(period_rate, n, pay):
        if period_rate == 0:
            return -pay * n
        return -pay * (1 - (1 + period_rate) ** (-n)) / period_rate

    fv0 = -pv(rate0, R, payment)
    fv1 = -pv(rate1, R, payment)

    return round(fv1 - fv0, 2)



class CreditGenerator:
    """
    This class gets ranges for principals, maturities, and credit spreads,
    and returns a dataframe with the delta fair value of the credits.
    """

    def __init__(self, credit_types, principals_ranges, maturities_ranges,
                 distributions, credit_spread_ranges, num_credits,
                 interest_rates, random_seed=42):
        self.credit_types = credit_types
        self.principals_ranges = principals_ranges
        self.maturities_ranges = maturities_ranges
        self.distributions = distributions
        self.credit_spread_ranges = credit_spread_ranges
        self.num_credits = num_credits
        self.interest_rates = interest_rates  # [rate0, rate1] for two months
        np.random.seed(random_seed)
        self.validate_parameters()

    def validate_parameters(self):
        num_types = len(self.credit_types)
        assert len(self.principals_ranges) == num_types, "Mismatch in principal ranges."
        assert len(self.maturities_ranges) == num_types, "Mismatch in maturity ranges."
        assert len(self.distributions) == num_types, "Mismatch in distributions."
        assert len(self.credit_spread_ranges) == num_types, "Mismatch in spread ranges."
        assert np.isclose(sum(self.distributions), 1.0), "Distributions must sum to 1."

    def get_credit_types_info(self):
        credit_types_info = []
        for i in range(len(self.credit_types)):
            credit_type_info = {
                'Type': self.credit_types[i],
                'Credit_Spread': self.credit_spread_ranges[i],
                'Principal': self.principals_ranges[i],
                'Maturity': self.maturities_ranges[i]
            }
            credit_types_info.append(credit_type_info)
        return credit_types_info

    def generate_credits(self):
        credits = []
        credit_types_info = self.get_credit_types_info()
        for _ in range(self.num_credits):
            credit_type_info = np.random.choice(credit_types_info, p=self.distributions)
            principal = np.random.randint(credit_type_info['Principal'][0],
                                          credit_type_info['Principal'][1])
            maturity = np.random.randint(credit_type_info['Maturity'][0],
                                         credit_type_info['Maturity'][1])
            spread_range = credit_type_info['Credit_Spread']
            credit_spread = np.random.uniform(spread_range[0], spread_range[1])

            delta_fv = credit_fv_delta(principal, maturity, credit_spread, self.interest_rates)

            credits.append({
                'Type': credit_type_info['Type'],
                'Principal': principal,
                'Maturity': maturity,
                'Credit_Spread': credit_spread,
                'Delta_FV': delta_fv
            })
        return pd.DataFrame(credits)
