import pandas as pd
import numpy as np
from collections import Counter
import re

def copypasta_score(matches, df, threshold=1):
    """
    Calculates the percentage of rows in `df` that appear in `matches` 
    at least `threshold` times.

    Parameters:
        matches (pd.DataFrame): Must contain 'source' and 'target' columns.
        df (pd.DataFrame): Original DataFrame to evaluate.
        df_integral (object): Object with a `.name` attribute used as a key in the result.
        threshold (int): Minimum number of occurrences to consider a match.

    Returns:
        dict: {'tweet_in_cluster (%)': value}
    """
    all_matches = list(matches['source']) + list(matches['target'])
    match_counts = Counter(all_matches)
    counts_series = df.index.to_series().map(match_counts).fillna(0)
    mask = counts_series >= threshold
    score = len(df[mask]) / len(df) * 100

    return score


def get_top_users(df):
    """
    Function to calculate the percentage of posts from the most active users.
    """
    user_post_counts = df['username'].value_counts()

    # Calculate 10% of the total number of users (rounded up)
    top_10_percent_count = int(np.ceil(len(user_post_counts) * 0.01))

    # Extract the top users
    top_users = user_post_counts.head(top_10_percent_count)

    # Filter the original dataframe to include only these users
    top_users_df = df[df['username'].isin(top_users.index)]

    len(top_users_df) / len(df) * 100

    return len(top_users_df) / len(df) * 100 , top_10_percent_count


def calculate_zero_fw_score(df: pd.DataFrame) -> tuple:
    """
    Calculate the percentage of rows where both 'followers' and 'following' are less than 1.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'followers', 'following',
                           and a 'name' attribute (either a value or a pd.Series).
    
    Returns:
        tuple: (name, zero_score_percent) where zero_score_percent is rounded to 4 decimal places.
    """

    mask = (df['followers'] < 1) & (df['following'] < 1)
    zero_score = len(df[mask]) / len(df) * 100 if len(df) > 0 else 0

    return zero_score

def no_image_description_score(df: pd.DataFrame) -> tuple:
    """
    Calculate the percentage of users with no bio and a default/empty/missing avatar.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'bio', 'avatar'
                           
    
    Returns:
        tuple: (name, no_image_description_percent) where the percent is rounded to 4 decimal places.
    """

    mask_desc = (df['bio'] == "") | (df['bio'].isna())

    mask_image = (
        (df['avatar'] == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png') |
        (df['avatar'] == '') |
        (df['avatar'].isna())
    )

    mask = mask_desc & mask_image
    zero_score = len(df[mask]) / len(df) * 100 if len(df) > 0 else 0

    return zero_score

def _check_username_digits(username: str, num_digits: int) -> bool:
    """
    Check if a Twitter username ends with a specified number of digits.
    
    Parameters:
        username (str): The username string to check.
        num_digits (int): The number of digits the username should end with.
    
    Returns:
        bool: True if the username ends with the specified number of digits, else False.
    """
    pattern = r'\d{' + str(num_digits) + r'}$'
    return bool(re.search(pattern, username))

def default_handle_score(df: pd.DataFrame, num_digits: int = 5) -> tuple:
    """
    Calculate the percentage of usernames that end with a given number of digits.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with a 'username' column and a 'name' attribute.
        num_digits (int): The number of trailing digits to check in usernames (default is 5).
    
    Returns:
        tuple: (name, default_handle_percent) rounded to 2 decimal places.
    """

    matches = df['username'].apply(lambda x: _check_username_digits(x, num_digits))
    if matches.any():
        n_true = matches.value_counts(normalize=True).get(True, 0) * 100
    else:
        n_true = 0.0

    return n_true