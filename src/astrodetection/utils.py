import pandas as pd
import numpy as np
from collections import Counter
import re
import networkx as nx
from networkx.algorithms import community
from ipysigma import Sigma
from typing import Iterable, Optional, Dict, Union

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

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
   # df_unique = df.drop_duplicates(subset=[username_col]).copy()

    mask = (df[followers_col] < 1) & (df[following_col] < 1)
    zero_score = len(df[mask]) / len(df) * 100 if len(df) > 0 else 0

    return zero_score

def no_image_description_score(df: pd.DataFrame, bio_col: str = 'bio', avatar_col: str = 'avatar', username_col: str = 'username') -> float:
    """
    Calculate the percentage of users with no bio and a default/empty/missing avatar.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'bio', 'avatar'
                           
    
    Returns:
        tuple: (name, no_image_description_percent) where the percent is rounded to 4 decimal places.
    """

    #df_unique = df.drop_duplicates(subset=[username_col]).copy()

    mask_desc = (df[bio_col] == "") | (df[bio_col].isna())

    mask_image = (
        (df[avatar_col] == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png') |
        (df[avatar_col] == '') |
        (df[avatar_col].isna())
    )

    mask = mask_desc & mask_image
    zero_score = len(df[mask]) / len(df) * 100 if len(df) > 0 else 0

    return zero_score

def over_tot_post_per_day(df: pd.DataFrame, threshold: int = 70, tweets_per_day_col: str = 'tweets_per_day', username_col: str = 'username') -> float:
    """
    Calculate the percentage of users that post more than a specified number of posts per day.

    Parameters:
        df (pd.DataFrame): The input DataFrame, which must have 'tweet_per_day' column
        threshold (int): The minimum number of posts per day to consider. Default is 70
    """
    #df_unique = df.drop_duplicates(subset=[username_col]).copy()
    mask = df[tweets_per_day_col] > threshold
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
    #df_unique = df.drop_duplicates(subset=[username_col]).copy()

    matches = df[username_col].apply(lambda x: _check_username_digits(x, num_digits))
    if matches.any():
        n_true = matches.value_counts(normalize=True).get(True, 0) * 100
    else:
        n_true = 0.0

    return n_true

def check_recent_account(
    df: pd.DataFrame, 
    account_creation_col: str = 'createdDate',
    tweet_date_col: str = 'tweet_date',
    age_days_threshold: int = 30,
    username_col: str = 'username'
    ):
    """
    Calculate the percentage of accounts created within a certain number of days before the tweet date.
    """
    #df_unique = df.drop_duplicates(subset=[username_col]).copy()
    df[tweet_date_col] = pd.to_datetime(df[tweet_date_col]).dt.tz_localize(None)
    df[account_creation_col] = pd.to_datetime(df[account_creation_col]).dt.tz_localize(None)

    delta_days = (df[tweet_date_col] - df[account_creation_col]).dt.days

    recent_accounts = delta_days < age_days_threshold

    proportion = len(df[recent_accounts]) / len(df) * 100 if len(df) > 0 else 0

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
    #df_unique = df.drop_duplicates(subset=[username_col]).copy()
    creation_weeks = df[account_creation_col].dt.to_period('W')
    week_counts = creation_weeks.value_counts()

    return week_counts[:n_weeks].sum() / len(df) * 100 if len(df) > 0 else 0

def _check_excessive_tags(tweet, threshold=4):
    """
    Check if a tweet has more than a specified number of people tagged.
    
    Args:
        tweet (str): The tweet text to analyze.
        threshold (int): Maximum number of tags allowed (default: 4).
        
    Returns:
        bool: True if the tweet has more than threshold tags, False otherwise.
    """
    import re
    # Find all @mentions in the tweet using Twitter's username rules
    # Pattern matches @ followed by 1-15 alphanumeric characters or underscores
    # (?<!\w) ensures @ is not preceded by a word character
    tags = re.findall(r'(?<!\w)@([a-z0-9_]{1,15})', tweet.lower())
    
    return len(tags) > threshold

def excessive_tags_score(
    df: pd.DataFrame, 
    tweet_text_col: str = 'tweet'):
    """
    Calculate the excessive tags score for a DataFrame of tweets.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets.
        tweet_text_col (str): Name of the column with tweet texts.
        
    Returns:
        float: Proportion of tweets with excessive tags.
    """
    df = df.dropna(subset=[tweet_text_col])
    excessive_tagged = df[tweet_text_col].apply(_check_excessive_tags)
    score = excessive_tagged.sum() / len(df) * 100
    return score


#find communities in G_sharing and save them as node attributes for later use in Gephi

def get_similarity_hub_score(G_sharing, df, threshold=0.9, username_col="screen name", type_col="row_type"):
    """
    Compute the similarity hub score based on the largest community in the co-sharing graph.

    The score represents the proportion of retweeting users that belong to the largest
    connected community after filtering edges below a similarity threshold. A higher score
    indicates a more concentrated coordinated-activity hub.

    Args:
        G_sharing (nx.Graph): Co-sharing similarity graph returned by coSharing().
            Edges must have a 'weight' attribute representing cosine similarity.
        df (pd.DataFrame): DataFrame containing retweet event rows. Used to count the total number of unique retweeting users.
        threshold (float): Minimum edge weight to retain. Edges with weight < threshold
            are removed before community detection. Default is 0.9.
        username_col (str): Name of the column in df that holds user identifiers.
            Default is "screen name".
        type_col (str or None): Name of the column in df that identifies row type.
            If provided, only rows where type_col == 'retweet' are counted as users.
            If None, all rows are counted. Default is "row_type".

    Returns:
        float: The hub score as a percentage (0–100). Calculated as:
            (size of largest community / total unique retweeting users) * 100.
            Returns 0 if there are no retweeting users.
    """

    G_sharing = G_sharing.copy()  # avoid modifying the original graph

    edges_to_remove = [(u, v) for u, v, d in G_sharing.edges(data=True) if d.get('weight', 0) < threshold]

    G_sharing.remove_edges_from(edges_to_remove)
    G_sharing.remove_nodes_from(list(nx.isolates(G_sharing)))

    communities = community.greedy_modularity_communities(G_sharing, weight='weight')
    

    largest_community_size = len(communities[0])

    if type_col:
        n_users = df[df[type_col]=='retweet'][username_col].nunique()
    else:
        n_users = df[username_col].nunique()

    score = largest_community_size / n_users if n_users > 0 else 0
    return score * 100


def compute_bot_likelihood_metrics(
    df: pd.DataFrame,
    matches: pd.DataFrame = None,
    matches_threshold: int = 1,
    num_digits: int = 5,
    top_x_percent: int = 1,
    over_post_per_day_threshold: int = 70,
    age_days_threshold: int = 30,
    n_weeks: int = 4,
    G_sharing: nx.Graph = None,
    similarity_sharing_threshold: float = 0.9,
    # Column name overrides (keep defaults for backwards compatibility)
    username_col: str = 'username',
    followers_col: str = 'followers',
    following_col: str = 'following',
    bio_col: str = 'bio',
    avatar_col: str = 'avatar',
    tweets_per_day_col: str = 'tweets_per_day',
    account_creation_col: str = 'createdDate',
    tweet_date_col: str = 'tweet_date',
    tweet_text_col: str = 'tweet',
    type_col: str = 'row_type',

) -> dict:
    """
    Combine multiple behavioral metrics to estimate the likelihood that a set of accounts consists of bots
    or is engaged in coordinated inauthentic behavior.

    Computes up to 10 indicators. Each metric is included in the result dict only when the required
    columns are present in `df` (or the required arguments are provided); otherwise its value is None.

    Parameters:
        df (pd.DataFrame): Main DataFrame containing account/post data.
        matches (pd.DataFrame, optional): DataFrame with 'source' and 'target' columns (output of
            `semantic_faiss`). When provided, the copypasta score is computed.
        matches_threshold (int): Minimum number of match occurrences to count a post as duplicated
            for the copypasta score. Default is 1.
        num_digits (int): Number of trailing digits in a username that qualifies it as a
            "default handle" (e.g. auto-generated). Default is 5.
        top_x_percent (int): Percentage of most-active users to consider for the top-user dominance
            metric (e.g. 1 => top 1%). Default is 1.
        over_post_per_day_threshold (int): Minimum tweets-per-day value above which a user is
            considered an over-poster. Default is 70.
        age_days_threshold (int): Maximum account age in days (at time of tweet) to classify an
            account as recently created. Default is 30.
        n_weeks (int): Number of top account-creation weeks used to compute the creation-week
            cluster metric. Default is 4.
        G_sharing (nx.Graph, optional): Co-sharing similarity graph (output of
            `create_coSharing_graph`). Required to compute the similarity hub score.
        similarity_sharing_threshold (float): Minimum edge weight to retain when filtering
            `G_sharing` before community detection. Default is 0.9.
        username_col (str): Column name for user handles. Default is 'username'.
        followers_col (str): Column name for follower count. Default is 'followers'.
        following_col (str): Column name for following count. Default is 'following'.
        bio_col (str): Column name for user biography/description. Default is 'bio'.
        avatar_col (str): Column name for avatar URL. Default is 'avatar'.
        tweets_per_day_col (str): Column name for average tweets per day. Default is 'tweets_per_day'.
        account_creation_col (str): Column name for account creation date. Default is 'createdDate'.
        tweet_date_col (str): Column name for tweet/post date. Default is 'tweet_date'.
        tweet_text_col (str): Column name for tweet text. Default is 'tweet'.
        type_col (str): Column name identifying the row type (e.g. 'retweet'). Used by the
            similarity hub score to filter retweeting users. Default is 'row_type'.

    Returns:
        dict: Dictionary with the following keys (value is None when data is unavailable):
            - 'copypasta_score (%)': % of posts appearing in at least `matches_threshold` duplicate pairs.
            - 'top_users_post_percent (%)': % of posts authored by the top `top_x_percent`% of users.
            - 'top_users_count': Absolute number of users in the top-percent group.
            - 'zero_followers_and_following (%)': % of rows where both followers and following < 1.
            - 'no_image_and_description (%)': % of rows with an empty bio and a default/missing avatar.
            - 'default_handle_score (%)': % of usernames ending with `num_digits` or more digits.
            - 'over_tweet_per_day (%)': % of rows where tweets_per_day exceeds `over_post_per_day_threshold`.
            - 'recent_account_creation (%)': % of rows where account age at tweet time < `age_days_threshold` days.
            - 'top_creation_weeks (%)': % of accounts created in the top `n_weeks` most common creation weeks.
            - 'excessive_tags_score (%)': % of tweets mentioning more than 4 users (@tags).
            - 'similarity_hub_score (%)': % of retweeting users belonging to the largest community in `G_sharing`.
            - 'number_of_tweets': Total number of rows in `df`.
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
    
    # 8. Excessive Tags Score
    if tweet_text_col in df.columns:
        results['excessive_tags_score (%)'] = round(excessive_tags_score(df, tweet_text_col=tweet_text_col), 2)
    else:
        results['excessive_tags_score (%)'] = None
    
    # 9. Similarity Hub Score
    if G_sharing is not None and username_col in df.columns:
        results['similarity_hub_score (%)'] = round(get_similarity_hub_score(G_sharing, df, threshold=similarity_sharing_threshold, username_col=username_col, type_col=type_col), 2)
    else:
        results['similarity_hub_score (%)'] = None

    # 10. Support number of tweets
    results['number_of_tweets'] = len(df)

    return results