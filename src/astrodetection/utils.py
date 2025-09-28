import pandas as pd
import numpy as np
from collections import Counter
import re
import networkx as nx
from ipysigma import Sigma
from typing import Iterable, Optional, Dict, Union

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


def get_top_users(df, percent=1, username_col: str = 'username'):
    """Calculate the percentage of posts coming from the top *percent* most active users.

    Parameters:
        df (pd.DataFrame)
        percent (float|int): Percentage of users to consider (e.g. 1 => top 1%).
        username_col (str): Column name containing the user handle. Defaults to 'username'.

    Returns:
        (posts_percent, top_users_count)
    """
    user_post_counts = df[username_col].value_counts()

    ratio = percent / 100

    # Calculate x% of the total number of users (rounded up)
    top_x_percent_count = int(np.ceil(len(user_post_counts) * ratio))

    # Extract the top users
    top_users = user_post_counts.head(top_x_percent_count)

    # Filter the original dataframe to include only these users
    top_users_df = df[df[username_col].isin(top_users.index)]

    len(top_users_df) / len(df) * 100

    return len(top_users_df) / len(df) * 100 , top_x_percent_count


def calculate_zero_fw_score(df: pd.DataFrame, followers_col: str = 'followers', following_col: str = 'following', username_col: str = 'username') -> float:
    """
    Calculate the percentage of rows where both 'followers' and 'following' are less than 1.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'followers', 'following',
                           and a 'name' attribute (either a value or a pd.Series).
    
    Returns:
        tuple: (name, zero_score_percent) where zero_score_percent is rounded to 4 decimal places.
    """
    df_unique = df.drop_duplicates(subset=[username_col]).copy()
    mask = (df_unique[followers_col] < 1) & (df_unique[following_col] < 1)
    zero_score = len(df_unique[mask]) / len(df_unique) * 100 if len(df_unique) > 0 else 0

    return zero_score

def no_image_description_score(df: pd.DataFrame, bio_col: str = 'bio', avatar_col: str = 'avatar', username_col: str = 'username') -> float:
    """
    Calculate the percentage of users with no bio and a default/empty/missing avatar.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'bio', 'avatar'
                           
    
    Returns:
        tuple: (name, no_image_description_percent) where the percent is rounded to 4 decimal places.
    """

    df_unique = df.drop_duplicates(subset=[username_col]).copy()

    mask_desc = (df_unique[bio_col] == "") | (df_unique[bio_col].isna())

    mask_image = (
        (df_unique[avatar_col] == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png') |
        (df_unique[avatar_col] == '') |
        (df_unique[avatar_col].isna())
    )

    mask = mask_desc & mask_image
    zero_score = len(df_unique[mask]) / len(df_unique) * 100 if len(df_unique) > 0 else 0

    return zero_score

def over_tot_post_per_day(df: pd.DataFrame, threshold: int = 70, tweets_per_day_col: str = 'tweets_per_day', username_col: str = 'username') -> float:
    """
    Calculate the percentage of users that post more than a specified number of posts per day.

    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'tweet_per_day' column
        threshold (int): The minimum number of posts per day to consider. Default is 70
    """
    df_unique = df.drop_duplicates(subset=[username_col]).copy()
    mask = df_unique[tweets_per_day_col] > threshold
    return mask.value_counts(normalize=True).get(True, 0) * 100

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

def default_handle_score(df: pd.DataFrame, num_digits: int = 5, username_col: str = 'username') -> float:
    """
    Calculate the percentage of usernames that end with a given number of digits.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with a 'username' column and a 'name' attribute.
        num_digits (int): The number of trailing digits to check in usernames (default is 5).
    
    Returns:
        tuple: (name, default_handle_percent) rounded to 2 decimal places.
    """
    df_unique = df.drop_duplicates(subset=[username_col]).copy()

    matches = df_unique[username_col].apply(lambda x: _check_username_digits(x, num_digits))
    if matches.any():
        n_true = matches.value_counts(normalize=True).get(True, 0) * 100
    else:
        n_true = 0.0

    return n_true

def check_recent_account(
    df: pd.DataFrame, 
    account_creation_col: str = 'createdDate',
    tweet_date_col: str = 'tweet_date',
    age_days_threshold: int = 1000,
    username_col: str = 'username'
    ):
    """
    Calculate the percentage of accounts created within a certain number of days before the tweet date.
    """
    df_unique = df.drop_duplicates(subset=[username_col]).copy()
    df_unique[tweet_date_col] = pd.to_datetime(df_unique[tweet_date_col]).dt.tz_localize(None)
    df_unique[account_creation_col] = pd.to_datetime(df_unique[account_creation_col]).dt.tz_localize(None)

    delta_days = (df_unique[tweet_date_col] - df_unique[account_creation_col]).dt.days

    recent_accounts = delta_days < age_days_threshold

    proportion = len(df_unique[recent_accounts]) / len(df_unique) * 100 if len(df_unique) > 0 else 0

    return proportion

def check_creation_week_cluster(
    df: pd.DataFrame, 
    account_creation_col: str = 'createdDate', 
    n_weeks: int = 4,
    username_col: str = 'username'
) -> float:
    """
    Calculate the percentage of accounts created within the top N most common account creation weeks.
    """
    df_unique = df.drop_duplicates(subset=[username_col]).copy()
    creation_weeks = df_unique[account_creation_col].dt.to_period('W')
    week_counts = creation_weeks.value_counts()

    return week_counts[:n_weeks].sum() / len(df_unique) * 100 if len(df_unique) > 0 else 0



def compute_bot_likelihood_metrics(
    df: pd.DataFrame,
    matches: pd.DataFrame = None,
    matches_threshold: int = 1,
    num_digits: int = 5,
    top_x_percent: int = 1,
    over_post_per_day_threshold: int = 70,
    age_days_threshold: int = 1000,
    n_weeks: int = 4,
    # Column name overrides (keep defaults for backwards compatibility)
    username_col: str = 'username',
    followers_col: str = 'followers',
    following_col: str = 'following',
    bio_col: str = 'bio',
    avatar_col: str = 'avatar',
    tweets_per_day_col: str = 'tweets_per_day',
    account_creation_col: str = 'createdDate',
    tweet_date_col: str = 'tweet_date',
) -> dict:
    """
    Combina diverse metriche per stimare la probabilità che un insieme di account sia composto da bot.

    Parameters:
        df (pd.DataFrame): Il DataFrame principale contenente i dati degli account/post.
        matches (pd.DataFrame, optional): DataFrame con colonne 'source' e 'target' per il punteggio copypasta.
        matches_threshold (int): Threshold minimo di occorrenze per considerare un match nel punteggio copypasta.
        num_digits (int): Numero di cifre finali nel nome utente per rilevare handle predefiniti.
        over_post_per_day_threshold (int): Soglia per il numero di post al giorno per considerare un utente come "over".

    Returns:
        dict: Dizionario con tutte le metriche calcolate.
    """

    results = {}

    # 1. Copypasta Score (solo se fornito `matches`)
    if matches is not None:
        results['copypasta_score (%)'] = round(copypasta_score(matches, df, matches_threshold), 2)

    # 2. Top User Dominance
    top_users_percent, top_users_n = get_top_users(df, top_x_percent, username_col=username_col)
    results['top_users_post_percent (%)'] = round(top_users_percent, 2)
    results['top_users_count'] = top_users_n

    # 3. Zero Followers & Following
    # 3. Zero Followers & Following
    if followers_col in df.columns and following_col in df.columns:
        results['zero_followers_and_following (%)'] = round(calculate_zero_fw_score(df, followers_col=followers_col, following_col=following_col, username_col=username_col), 2)
    else:
        results['zero_followers_and_following (%)'] = None

    # 4. No Image and Description
    if bio_col in df.columns and avatar_col in df.columns:
        results['no_image_and_description (%)'] = round(no_image_description_score(df, bio_col=bio_col, avatar_col=avatar_col, username_col=username_col), 2)
    else:
        results['no_image_and_description (%)'] = None

    # 5. Default Handle Score
    if username_col in df.columns:
        results['default_handle_score (%)'] = round(default_handle_score(df, num_digits, username_col=username_col), 2)
    else:
        results['default_handle_score (%)'] = None

    if tweets_per_day_col in df.columns:
        results['over_tweet_per_day (%)'] = round(over_tot_post_per_day(df, over_post_per_day_threshold, tweets_per_day_col=tweets_per_day_col, username_col=username_col), 2)
    else:
        results['over_tweet_per_day (%)'] = None

    # 6. Recent Account Creation
    if account_creation_col in df.columns and tweet_date_col in df.columns:
        results['recent_account_creation (%)'] = round(check_recent_account(df, account_creation_col=account_creation_col, tweet_date_col=tweet_date_col, age_days_threshold=age_days_threshold, username_col=username_col), 2)
    else:
        results['recent_account_creation (%)'] = None
    
    # 7. Account Creation weeks clusters
    if account_creation_col in df.columns:
        results['top_creation_weeks (%)'] = round(check_creation_week_cluster(df, account_creation_col=account_creation_col, n_weeks=n_weeks, username_col=username_col), 2)
    else:
        results['top_creation_weeks (%)'] = None

    # 8. Support number of tweets
    results['number_of_tweets'] = len(df)

    return results

def create_network(
    match_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    # metadata_df overridable column names (match_df assumed to have default columns)
    username_col: str = 'username',
    likes_col: str = 'likes_count',
    tweet_date_col: str = 'tweet_date',
    link_col: str = 'link_tweet',
    label: str = 'text',
    # NEW: flexible extra metadata controls
    extra_meta: Union[str, Iterable[str]] = None,           # "*" => include all other columns
    exclude_meta: Optional[Iterable[str]] = None,          # columns to skip
    extra_meta_prefix: Optional[str] = None,               # e.g., "meta_"
    extra_meta_rename: Optional[Dict[str, str]] = None,    # rename extras {old:new}
    keep_na: bool = False                                  # drop None/NaN extras by default
):
    """
    Create a directed graph representing tweet relationships and metadata.

    Parameters
    ----------
    match_df : pd.DataFrame
        REQUIRED columns (fixed schema):
            - 'source', 'target', 'text_to_embed_source', 'text_to_embed_target', 'score'
            - 'dup_type' (optional)

    metadata_df : pd.DataFrame
        Indexed by tweet ID. May include:
            - username_col (default 'username')
            - likes_col (default 'likes_count')
            - tweet_date_col (default 'tweet_date')
            - link_col (default 'link_tweet')
        Missing columns or index entries are tolerated and replaced with None.

    label : {'text','author'}, default 'text'
        Node label selection. If 'author', tweet text is stored under key 'text'.

    Flexible extras
    ---------------
    extra_meta : {'*' or iterable of column names}, default None
        Include arbitrary extra columns from metadata_df as node attributes.
    exclude_meta : iterable of column names to exclude from extras.
    extra_meta_prefix : optional prefix for extra keys to avoid name collisions.
    extra_meta_rename : dict mapping {original_column: new_key_name}.
    keep_na : bool, default False
        If False, drop extras whose value is None/NaN.

    Returns
    -------
    Sigma
        A Sigma visualization object representing the network.
    """
    graph = nx.DiGraph()

    if label not in {"text", "author"}:
        raise ValueError("label must be either 'text' or 'author'")

    # Helper: safely pull a scalar value from metadata_df for a node/column
    def _safe_meta(node_id, col_name):
        if col_name in metadata_df.columns and node_id in metadata_df.index:
            value = metadata_df.loc[node_id, col_name]
            # Handle duplicate index -> Series/DataFrame cases
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            return value
        return None

    # Helper: safely pull a whole row (Series) for extras
    def _safe_row(node_id) -> pd.Series:
        if node_id not in metadata_df.index:
            return pd.Series(dtype=object)
        row = metadata_df.loc[node_id]
        if isinstance(row, pd.DataFrame):  # duplicated index -> take first row
            row = row.iloc[0]
        return row

    # Decide which extra columns to include
    std_cols = {username_col, likes_col, tweet_date_col, link_col}
    exclude_meta = set(exclude_meta or [])
    rename_map = dict(extra_meta_rename or {})

    if extra_meta == "*":
        candidate_cols = set(metadata_df.columns) - std_cols
    elif extra_meta is None:
        candidate_cols = set()
    else:
        candidate_cols = set(extra_meta) - std_cols

    candidate_cols -= exclude_meta
    # Ensure we don't collide with existing keys we set explicitly
    reserved_keys = {"label", "author", "text", "likes", "time", "link"}

    def _extract_extras(node_id) -> Dict[str, object]:
        row = _safe_row(node_id)
        if row.empty:
            return {}
        extras = {}
        for col in candidate_cols:
            if col not in row.index:
                continue
            v = row[col]
            # Normalize pandas-y values
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif pd.isna(v):
                v = None
            key = rename_map.get(col, col)
            if extra_meta_prefix:
                key = f"{extra_meta_prefix}{key}"
            # Avoid collisions with our reserved keys
            if key in reserved_keys:
                key = f"extra_{key}"
            if v is None and not keep_na:
                continue
            extras[key] = v
        return extras

    for _, r in match_df.iterrows():
        source_id = r['source']
        target_id = r['target']

        source_author = _safe_meta(source_id, username_col)
        target_author = _safe_meta(target_id, username_col)
        source_text = r['text_to_embed_source']
        target_text = r['text_to_embed_target']

        if label == 'text':
            source_data = {
                "label": source_text,
                "author": source_author,
                "likes": _safe_meta(source_id, likes_col),
                "time": _safe_meta(source_id, tweet_date_col),
                "link": _safe_meta(source_id, link_col),
            }
            target_data = {
                "label": target_text,
                "author": target_author,
                "likes": _safe_meta(target_id, likes_col),
                "time": _safe_meta(target_id, tweet_date_col),
                "link": _safe_meta(target_id, link_col),
            }
        else:  # label == 'author'
            source_data = {
                "label": source_author,
                "author": source_author,
                "text": source_text,
                "likes": _safe_meta(source_id, likes_col),
                "time": _safe_meta(source_id, tweet_date_col),
                "link": _safe_meta(source_id, link_col),
            }
            target_data = {
                "label": target_author,
                "author": target_author,
                "text": target_text,
                "likes": _safe_meta(target_id, likes_col),
                "time": _safe_meta(target_id, tweet_date_col),
                "link": _safe_meta(target_id, link_col),
            }

        # Merge in extra metadata (arbitrary columns)
        source_data.update(_extract_extras(source_id))
        target_data.update(_extract_extras(target_id))

        # Add nodes/edges
        graph.add_node(source_id, **source_data)
        graph.add_node(target_id, **target_data)

        graph.add_edge(
            source_id,
            target_id,
            dup_type=r.get('dup_type', "default"),
            weight=r['score'],
        )

    sigma_viz = Sigma(
        graph,
        edge_color="dup_type",
        edge_weight="weight",
        node_size="likes",
        node_size_range=(3, 15),
    )
    return sigma_viz