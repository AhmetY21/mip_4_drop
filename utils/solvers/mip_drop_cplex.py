from docplex.mp.model import Model
import pandas as pd
from cplex.exceptions import CplexSolverError

import os

cplex_status_mapping = {
        1: "CPX_STAT_OPTIMAL",
        2: "CPX_STAT_UNBOUNDED",
        3: "CPX_STAT_INFEASIBLE",
        4: "CPX_STAT_INForUNBD",
        5: "CPX_STAT_OPTIMAL_INFEAS",
        6: "CPX_STAT_NUM_BEST",
        10: "CPX_STAT_ABORT_IT_LIM",
        11: "CPX_STAT_ABORT_TIME_LIM",
        12: "CPX_STAT_ABORT_OBJ_LIM",
        13: "CPX_STAT_ABORT_USER",
        14: "CPX_STAT_FEASIBLE_RELAXED_SUM",
        15: "CPX_STAT_OPTIMAL_RELAXED_SUM",
        16: "CPX_STAT_FEASIBLE_RELAXED_INF",
        17: "CPX_STAT_OPTIMAL_RELAXED_INF",
        18: "CPX_STAT_FEASIBLE_RELAXED_QUAD",
        19: "CPX_STAT_OPTIMAL_RELAXED_QUAD",
        20: "CPX_STAT_OPTIMAL_FACE_UNBOUNDED",
        21: "CPX_STAT_ABORT_PRIM_OBJ_LIM",
        22: "CPX_STAT_ABORT_DUAL_OBJ_LIM",
        23: "CPX_STAT_FEASIBLE",
        24: "CPX_STAT_FIRSTORDER",
        25: "CPX_STAT_ABORT_DETTIME_LIM",
        30: "CPX_STAT_CONFLICT_FEASIBLE",
        31: "CPX_STAT_CONFLICT_MINIMAL",
        32: "CPX_STAT_CONFLICT_ABORT_CONTRADICTION",
        33: "CPX_STAT_CONFLICT_ABORT_TIME_LIM",
        34: "CPX_STAT_CONFLICT_ABORT_IT_LIM",
        35: "CPX_STAT_CONFLICT_ABORT_NODE_LIM",
        36: "CPX_STAT_CONFLICT_ABORT_OBJ_LIM",
        37: "CPX_STAT_CONFLICT_ABORT_MEM_LIM",
        38: "CPX_STAT_CONFLICT_ABORT_USER",
        39: "CPX_STAT_CONFLICT_ABORT_DETTIME_LIM",
        41: "CPX_STAT_BENDERS_NUM_BEST",
        101: "CPXMIP_OPTIMAL",
        102: "CPXMIP_OPTIMAL_TOL",
        103: "CPXMIP_INFEASIBLE",
        104: "CPXMIP_SOL_LIM",
        105: "CPXMIP_NODE_LIM_FEAS",
        106: "CPXMIP_NODE_LIM_INFEAS",
        107: "CPXMIP_TIME_LIM_FEAS",
        108: "CPXMIP_TIME_LIM_INFEAS",
        109: "CPXMIP_FAIL_FEAS",
        110: "CPXMIP_FAIL_INFEAS",
        111: "CPXMIP_MEM_LIM_FEAS",
        112: "CPXMIP_MEM_LIM_INFEAS",
        113: "CPXMIP_ABORT_FEAS",
        114: "CPXMIP_ABORT_INFEAS",
        115: "CPXMIP_OPTIMAL_INFEAS",
        116: "CPXMIP_FAIL_FEAS_NO_TREE",
        117: "CPXMIP_FAIL_INFEAS_NO_TREE",
        118: "CPXMIP_UNBOUNDED",
        119: "CPXMIP_INForUNBD",
        120: "CPXMIP_FEASIBLE_RELAXED_SUM",
        121: "CPXMIP_OPTIMAL_RELAXED_SUM",
        122: "CPXMIP_FEASIBLE_RELAXED_INF",
        123: "CPXMIP_OPTIMAL_RELAXED_INF",
        124: "CPXMIP_FEASIBLE_RELAXED_QUAD",
        125: "CPXMIP_OPTIMAL_RELAXED_QUAD",
        126: "CPXMIP_ABORT_RELAXED",
        127: "CPXMIP_FEASIBLE",
        128: "CPXMIP_POPULATESOL_LIM",
        129: "CPXMIP_OPTIMAL_POPULATED",
        130: "CPXMIP_OPTIMAL_POPULATED_TOL",
        131: "CPXMIP_DETTIME_LIM_FEAS",
        132: "CPXMIP_DETTIME_LIM_INFEAS",
        133: "CPXMIP_ABORT_RELAXATION_UNBOUNDED",
        301: "CPX_STAT_MULTIOBJ_OPTIMAL",
        302: "CPX_STAT_MULTIOBJ_INFEASIBLE",
        303: "CPX_STAT_MULTIOBJ_INForUNBD",
        304: "CPX_STAT_MULTIOBJ_UNBOUNDED",
        305: "CPX_STAT_MULTIOBJ_NON_OPTIMAL",
        306: "CPX_STAT_MULTIOBJ_STOPPED"
    }



    
'''    # Prepare the assignment matrix
    assignment = pd.DataFrame(index=credits_df.index, columns=swaps_df.index, data=0)
    for i in range(num_credits):
        for j in range(num_swaps):
            assignment.loc[i, j] = solution[x[i, j]]
    
    return assignment, solution[delta], status, mdl'''
from decimal import Decimal, ROUND_HALF_UP
from docplex.mp.linear import LinearExpr

def round_to_precision(value, precision=4):
    """Round numeric values to a specified precision."""
    return float(Decimal(value).quantize(Decimal(f'1e-{precision}'), rounding=ROUND_HALF_UP))

def round_linear_expr(expr, model, precision=4):
    """Rounds the coefficients in a linear expression to a specified precision."""
    if isinstance(expr, LinearExpr):
        # Create a new LinearExpr associated with the same model
        rounded_expr = LinearExpr(model)
        for var, coef in expr.iter_terms():
            rounded_coef = round_to_precision(coef, precision)
            rounded_expr.add_term(var, rounded_coef)
        return rounded_expr
    else:
        return round_to_precision(expr, precision)



def solve_with_cplex(credits_df: pd.DataFrame, swaps_df: pd.DataFrame,
                     verbose=False, time_limit=None, num_cpu=None,
                     mip_gap=0.01, rel_tol=1e-6, abs_tol=1e-6,
                     precision=4, export=True, experiment_name='First',
                     presolve=0, reduce=0):
    # Reset indices
    credits_df = credits_df.reset_index(drop=True)
    swaps_df = swaps_df.reset_index(drop=True)

    mdl = Model(name="Dollar_Offset_Optimization", log_output=verbose)

    if num_cpu is not None:
        mdl.context.cplex_parameters.threads = num_cpu
    if time_limit is not None:
        mdl.set_time_limit(time_limit)

    mdl.context.cplex_parameters.mip.tolerances.mipgap = mip_gap
    mdl.context.cplex_parameters.simplex.tolerances.feasibility = rel_tol
    mdl.context.cplex_parameters.simplex.tolerances.optimality = abs_tol
    mdl._float_precision = precision
    mdl.context.cplex_parameters.preprocessing.presolve = presolve
    mdl.context.cplex_parameters.preprocessing.reduce = reduce

    # Sizes
    num_credits = len(credits_df)
    num_swaps = len(swaps_df)

    # Variables
    delta = mdl.continuous_var(name="delta")
    x = mdl.binary_var_matrix(num_credits, num_swaps, name="x")

    mdl.minimize(delta)

    # Dollar offset constraints
    for j in range(num_swaps):
        lhs_var = mdl.continuous_var(name=f"lhs_{j}")
        rhs_upper_var = mdl.continuous_var(name=f"rhs_upper_{j}")
        rhs_lower_var = mdl.continuous_var(name=f"rhs_lower_{j}")

        lhs_expr = mdl.sum((-1)*credits_df.loc[i, 'Delta_FV'] * x[i, j] for i in range(num_credits))
        rhs_upper_expr = (1 + delta) * swaps_df.loc[j, 'Delta_FV']
        rhs_lower_expr = (1 - delta) * swaps_df.loc[j, 'Delta_FV']

        mdl.add_constraint(lhs_var == lhs_expr, ctname=f"lhs_def_{j}")
        mdl.add_constraint(rhs_upper_var == rhs_upper_expr, ctname=f"rhs_upper_def_{j}")
        mdl.add_constraint(rhs_lower_var == rhs_lower_expr, ctname=f"rhs_lower_def_{j}")

        mdl.add_constraint(lhs_var <= rhs_upper_var, ctname=f"Dollar_Offset_Upper_{j}")
        mdl.add_constraint(lhs_var >= rhs_lower_var, ctname=f"Dollar_Offset_Lower_{j}")

    # Each credit assigned at most once
    for i in range(num_credits):
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(num_swaps)) <= 1, ctname=f'credit_{i}_assignment')

    # Each swap gets at least one credit
    for j in range(num_swaps):
        mdl.add_constraint(mdl.sum(x[i, j] for i in range(num_credits)) >= 1, ctname=f'swap_{j}_assignment')

    # Sum of principals of assigned credits >= swap principal
    for j in range(num_swaps):
        mdl.add_constraint(mdl.sum(round(credits_df.loc[i, 'Principal'], precision) * x[i, j]
                                   for i in range(num_credits)) >=
                           round(swaps_df.loc[j, 'Principal'], precision),
                           ctname=f'Principal_Swap_{j}')

    # Weighted maturity constraint
    for j in range(num_swaps):
        maturity_lhs = mdl.sum(credits_df.loc[i, 'Principal'] * credits_df.loc[i, 'Maturity'] * x[i, j]
                               for i in range(num_credits))
        maturity_rhs = mdl.sum(credits_df.loc[i, 'Principal'] * x[i, j] for i in range(num_credits))
        swap_maturity = swaps_df.loc[j, 'Maturity']
        mdl.add_constraint(maturity_lhs >= swap_maturity * maturity_rhs, ctname=f'Maturity_Swap_{j}')

    # Export model
    base_output_folder = os.path.abspath(os.path.join(os.getcwd(), "../output/LP_Models"))
    os.makedirs(base_output_folder, exist_ok=True)
    model_file_path = os.path.join(base_output_folder, f"output_model_{experiment_name}.lp")
    if export:
        mdl.export_as_lp(model_file_path)

    solution = mdl.solve()
    print(mdl.solve_details.status)

    status = cplex_status_mapping[mdl.solve_details.status_code]
    assignment = pd.DataFrame(index=credits_df.index, columns=swaps_df.index, data=0)

    if solution is not None:
        print(f"Solution status: {status} with delta = {solution['delta']}")
        for i in range(num_credits):
            for j in range(num_swaps):
                assignment.loc[i, j] = int(round(solution[x[i, j]]))
        assignment.columns = [f"Credits_Assigned_Swap_{i}" for i in assignment.columns]
        return assignment, solution[delta], status, mdl
    else:
        print(f"Solution status: {status}")
        return None, None, status, mdl


