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

def compute_bot_likelihood_metrics(df: pd.DataFrame, matches: pd.DataFrame = None, threshold: int = 1, num_digits: int = 5) -> dict:
    """
    Combina diverse metriche per stimare la probabilità che un insieme di account sia composto da bot.

    Parameters:
        df (pd.DataFrame): Il DataFrame principale contenente i dati degli account/post.
        matches (pd.DataFrame, optional): DataFrame con colonne 'source' e 'target' per il punteggio copypasta.
        threshold (int): Threshold minimo di occorrenze per considerare un match nel punteggio copypasta.
        num_digits (int): Numero di cifre finali nel nome utente per rilevare handle predefiniti.

    Returns:
        dict: Dizionario con tutte le metriche calcolate.
    """

    results = {}

    # 1. Copypasta Score (solo se fornito `matches`)
    if matches is not None:
        results['copypasta_score (%)'] = round(copypasta_score(matches, df, threshold), 2)

    # 2. Top User Dominance
    top_users_percent, top_users_n = get_top_users(df)
    results['top_users_post_percent (%)'] = round(top_users_percent, 2)
    results['top_users_count'] = top_users_n

    # 3. Zero Followers & Following
    results['zero_followers_and_following (%)'] = round(calculate_zero_fw_score(df), 2)

    # 4. No Image and Description
    results['no_image_and_description (%)'] = round(no_image_description_score(df), 2)

    # 5. Default Handle Score
    results['default_handle_score (%)'] = round(default_handle_score(df, num_digits), 2)

    return results

def create_network(match_df, metadata_df):
    """
    Create a directed graph representing tweet relationships and metadata.

    Parameters:
    -----------
    match_df : pd.DataFrame
        DataFrame containing matched tweet pairs with columns:
            - 'source': ID of the source tweet
            - 'target': ID of the target tweet
            - 'text_to_embed_source': Text content of the source tweet
            - 'text_to_embed_target': Text content of the target tweet
            - 'score': Similarity score between source and target
            - 'dup_type' (optional): Type of duplication or relationship

    metadata_df : pd.DataFrame
        DataFrame indexed by tweet ID with metadata columns:
            - 'username': Author of the tweet
            - 'likes_count': Number of likes
            - 'tweet_date': Timestamp of the tweet
            - 'link_tweet': URL link to the tweet

    Returns:
    --------
    Sigma
        A Sigma visualization object representing the network.
    """

    graph = nx.DiGraph()

    for _, row in match_df.iterrows():
        source_id = row["source"]
        target_id = row["target"]

        # Extract source tweet metadata
        source_data = {
            "label": row["text_to_embed_source"],
            "author": metadata_df.loc[source_id, "username"],
            "likes": metadata_df.loc[source_id, "likes_count"],
            "time": metadata_df.loc[source_id, "tweet_date"],
            "link": metadata_df.loc[source_id, "link_tweet"]
        }

        # Extract target tweet metadata
        target_data = {
            "label": row["text_to_embed_target"],
            "author": metadata_df.loc[target_id, "username"],
            "likes": metadata_df.loc[target_id, "likes_count"],
            "time": metadata_df.loc[target_id, "tweet_date"],
            "link": metadata_df.loc[target_id, "link_tweet"]
        }

        # Add nodes to graph
        graph.add_node(source_id, **source_data)
        graph.add_node(target_id, **target_data)

        # Add directed edge with duplication type and similarity score
        graph.add_edge(
            source_id,
            target_id,
            dup_type=row.get("dup_type", "default"),
            weight=row["score"]
        )

    # Create Sigma visualization
    sigma_viz = Sigma(
        graph,
        edge_color="dup_type",
        edge_weight="weight",
        node_size="likes",
        node_size_range=(3, 15),
    )

    return sigma_viz