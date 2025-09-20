import pandas as pd
import numpy as np

def SwapGenerator(credits_df, num_swaps, fullfillment=0.9, random_factor=0.95):
    
    """
    Assigns credits to swaps and returns aggregated statistics for each swap.

    Parameters
    ----------
    credits_df : pd.DataFrame
        DataFrame containing credits with 'Principal', 'Delta_FV', and 'Maturity'.
    num_swaps : int
        Number of swaps to assign credits to.
    fullfillment : float, default=0.9
        The proportion of credits to retain for assignment.
    random_factor : float, default=1.0
        Factor to scale aggregated Delta_FV values (instead of internal randomness).

    Returns
    -------
    swap_agg_df : pd.DataFrame
        Aggregated statistics for each swap (Principal sum, Delta_FV sum, avg Maturity).
    adjusted_credits_df : pd.DataFrame
        All credits with their swap assignment (or Dropped).
    """
    # Retain credits according to fulfillment ratio
    retained_credits_df = credits_df.sample(frac=fullfillment, random_state=42)
    dropped_credits_df = credits_df.drop(retained_credits_df.index)

    # Randomly assign the retained credits to swaps
    retained_credits_df = retained_credits_df.copy()
    retained_credits_df['Swap'] = np.random.randint(1, num_swaps + 1, retained_credits_df.shape[0])

    # Aggregate by swap to get total Principal and Delta_FV
    swap_agg_df = retained_credits_df.groupby('Swap')[['Principal', 'Delta_FV']].sum().reset_index()

    # Weighted average maturity
    weighted_maturity_df = (
        retained_credits_df[['Swap', 'Principal', 'Maturity']]
        .assign(Weighted_Maturity=lambda x: x['Principal'] * x['Maturity'])
        .groupby('Swap')[['Weighted_Maturity', 'Principal']]
        .sum()
        .reset_index()
    )
    weighted_maturity_df['Maturity'] = weighted_maturity_df['Weighted_Maturity'] / weighted_maturity_df['Principal']
    weighted_maturity_df['Maturity'] = round(weighted_maturity_df['Maturity'],2)
    weighted_maturity_df.drop(columns=['Weighted_Maturity', 'Principal'], inplace=True)

    # Merge maturity into aggregated swaps
    swap_agg_df = swap_agg_df.merge(weighted_maturity_df, on='Swap')
    swap_agg_df["Principal"] = swap_agg_df["Principal"].astype(float)
    # Apply external random factor to Delta_FV
    swap_agg_df['Delta_FV'] = swap_agg_df['Delta_FV'] * random_factor * (-1)

    # Mark dropped credits
    dropped_credits_df = dropped_credits_df.copy()
    dropped_credits_df['Swap'] = 'Dropped'

    # Combine back
    adjusted_credits_df = pd.concat([retained_credits_df, dropped_credits_df], ignore_index=True)
    adjusted_credits_df['UNIQUE_INDEX'] = adjusted_credits_df.index

    return swap_agg_df, adjusted_credits_df
