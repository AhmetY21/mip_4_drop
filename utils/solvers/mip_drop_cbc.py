import os
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum, value, PULP_CBC_CMD

def solve_with_cbc(credits_df: pd.DataFrame,
                   swaps_df: pd.DataFrame,
                   verbose=False,
                   time_limit=None,
                   num_cpu=None,
                   mip_gap=0.01,
                   export=True,
                   experiment_name='First'):
    credits_df = credits_df.reset_index(drop=True).copy()
    swaps_df   = swaps_df.reset_index(drop=True).copy()

    for col in ['Principal', 'Delta_FV', 'Maturity']:
        credits_df[col] = credits_df[col].astype(float)
        swaps_df[col]   = swaps_df[col].astype(float)

    num_credits = len(credits_df)
    num_swaps   = len(swaps_df)

    mdl = LpProblem(name="Dollar_Offset_Optimization", sense=LpMinimize)

    delta = LpVariable('delta', lowBound=0.0)
    x = [[LpVariable(f'x_{i}_{j}', lowBound=0, upBound=1, cat='Binary')
          for j in range(num_swaps)] for i in range(num_credits)]

    mdl += delta

    for j in range(num_swaps):
        lhs = lpSum((-1) * credits_df.loc[i, 'Delta_FV'] * x[i][j] for i in range(num_credits))
        rhs_up = (1 + delta) * swaps_df.loc[j, 'Delta_FV']
        rhs_lo = (1 - delta) * swaps_df.loc[j, 'Delta_FV']
        mdl += lhs <= rhs_up, f"Dollar_Offset_Upper_{j}"
        mdl += lhs >= rhs_lo, f"Dollar_Offset_Lower_{j}"

    for i in range(num_credits):
        mdl += lpSum(x[i][j] for j in range(num_swaps)) <= 1, f'credit_{i}_assignment'

    for j in range(num_swaps):
        mdl += lpSum(x[i][j] for i in range(num_credits)) >= 1, f'swap_{j}_assignment'

    for j in range(num_swaps):
        mdl += lpSum(credits_df.loc[i, 'Principal'] * x[i][j] for i in range(num_credits)) \
               >= swaps_df.loc[j, 'Principal'], f'Principal_Swap_{j}'

    for j in range(num_swaps):
        numer = lpSum(credits_df.loc[i, 'Principal'] * credits_df.loc[i, 'Maturity'] * x[i][j]
                      for i in range(num_credits))
        denom = lpSum(credits_df.loc[i, 'Principal'] * x[i][j] for i in range(num_credits))
        mdl += numer >= swaps_df.loc[j, 'Maturity'] * denom, f'Maturity_Swap_{j}'

    solver = PULP_CBC_CMD(msg=verbose,
                          timeLimit=time_limit,
                          threads=num_cpu,
                          gapRel=mip_gap)

    if export:
        out_dir = os.path.abspath(os.path.join(os.getcwd(), "../output/LP_Models"))
        os.makedirs(out_dir, exist_ok=True)
        mdl.writeLP(os.path.join(out_dir, f"output_model_{experiment_name}.lp"))

    status = mdl.solve(solver)
    status_str = LpStatus[mdl.status]

    assignment = pd.DataFrame(index=credits_df.index, columns=swaps_df.index, data=0)
    if status_str in ['Optimal', 'Feasible']:
        delta_val = float(value(delta))
        for i in range(num_credits):
            for j in range(num_swaps):
                assignment.loc[i, j] = int(round(value(x[i][j])))
        assignment.columns = [f"Credits_Assigned_Swap_{j}" for j in assignment.columns]
        return assignment, delta_val, status_str, mdl
    else:
        return None, None, status_str, mdl
