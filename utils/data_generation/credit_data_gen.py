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
