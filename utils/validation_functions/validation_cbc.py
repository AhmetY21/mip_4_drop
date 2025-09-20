import pandas as pd
import numpy as np

def validate_solution_cbc(assignment: pd.DataFrame,
                          swaps_df: pd.DataFrame,
                          credits_df: pd.DataFrame,
                          objective_delta: float,
                          solver_name: str = None,
                          wall_time: float = None,
                          experiment_name: str = None):
    if 'UNIQUE_INDEX' in credits_df.columns and credits_df.index.name != 'UNIQUE_INDEX':
        credits_df = credits_df.set_index('UNIQUE_INDEX')
    if 'UNIQUE_INDEX' in assignment.columns and assignment.index.name != 'UNIQUE_INDEX':
        assignment = assignment.set_index('UNIQUE_INDEX')

    swap_cols = [c for c in assignment.columns if str(c).startswith('Credits_Assigned_Swap_')]
    swap_ids = [int(c.split('_')[-1]) for c in swap_cols]

    rows = []
    for col, j in zip(swap_cols, swap_ids):
        assigned_idx = assignment.index[assignment[col] == 1]
        assigned = credits_df.loc[credits_df.index.isin(assigned_idx)]

        assigned_principal = float(assigned['Principal'].sum()) if not assigned.empty else 0.0
        assigned_delta_fv  = float(assigned['Delta_FV'].sum()) if not assigned.empty else 0.0
        if assigned_principal > 0:
            assigned_wmaturity = float((assigned['Principal'] * assigned['Maturity']).sum() / assigned_principal)
        else:
            assigned_wmaturity = np.nan

        swap_row = swaps_df.loc[j]
        swap_principal = float(swap_row['Principal'])
        swap_delta_fv  = float(swap_row['Delta_FV'])
        swap_maturity  = float(swap_row['Maturity'])

        if np.isclose(swap_delta_fv, 0.0):
            r_j = np.nan
            delta_ok = np.isclose(assigned_delta_fv, 0.0)
        else:
            r_j = assigned_delta_fv / (-1 * swap_delta_fv)
            delta_ok = (0.85 <= r_j <= 1.15)

        principal_ok = (assigned_principal >= swap_principal)
        maturity_ok = (not np.isnan(assigned_wmaturity)) and (assigned_wmaturity >= swap_maturity)

        rows.append({
            'Swap_ID': j,
            'Assigned_Principal': round(assigned_principal, 2),
            'Swap_Principal': round(swap_principal, 2),
            'Principal_OK': bool(principal_ok),

            'Assigned_Delta_FV': round(assigned_delta_fv, 2),
            'Swap_Delta_FV': round(swap_delta_fv, 2),
            'r_j': (None if np.isnan(r_j) else round(r_j, 6)),
            'Delta_OK': bool(delta_ok),

            'Assigned_Weighted_Maturity': (None if np.isnan(assigned_wmaturity) else round(assigned_wmaturity, 6)),
            'Swap_Maturity': round(swap_maturity, 6),
            'Maturity_OK': bool(maturity_ok)
        })

    results_df = pd.DataFrame(rows).sort_values('Swap_ID').reset_index(drop=True)

    summary = {
        'Experiment_Name': experiment_name,
        'Solver': solver_name,
        'Wall_Time': wall_time,
        'Objective_Delta': f"{objective_delta:,.4f}",
        'All_Delta_OK': bool(results_df['Delta_OK'].all()) if not results_df.empty else False,
        'All_Principal_OK': bool(results_df['Principal_OK'].all()) if not results_df.empty else False,
        'All_Maturity_OK': bool(results_df['Maturity_OK'].all()) if not results_df.empty else False
    }

    return results_df, summary
