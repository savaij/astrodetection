from typing import Dict, Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ipysigma import Sigma


def create_coSharing_graph(data, type_column='row_type', userid_col='screen name', feature_col='retweeted user', min_retweets=2, min_overlap=3):
    """
    Build a co-sharing similarity graph among users based on shared retweet targets.

    Each node represents an active user (more than `min_retweets` total retweets). An edge between
    two users is added when they share at least `min_overlap` retweeted accounts in
    common, weighted by the TF-IDF cosine similarity of their retweet vectors.

    Algorithm steps:
        1. Filter to retweet rows only and keep accounts retweeted by more than one user.
        2. Count how many times each user retweeted each account (frequency matrix).
        3. Compute TF-IDF weights over the full user population so that IDF reflects
           the global popularity of each retweeted account.
        4. Restrict similarity computation to active users (> `min_retweets` total retweets).
        5. Compute pairwise cosine similarity among active users' TF-IDF vectors.
        6. Apply a hard overlap filter: zero out pairs sharing fewer than `min_overlap`
           distinct retweeted accounts.
        7. Build an undirected weighted graph from the resulting adjacency matrix and
           remove self-loops and isolated nodes.

    Args:
        data (pd.DataFrame): DataFrame containing retweet event rows, as produced by
            process_topic_data(). Must include columns for user id, retweeted account,
            and row type.
        type_column (str): Name of the column that identifies the row type. Only rows
            where this column equals 'retweet' are used. Default is 'row_type'.
        userid_col (str): Name of the column containing retweeting user identifiers.
            Default is 'screen name'.
        feature_col (str): Name of the column containing the retweeted account identifier.
            Default is 'retweeted user'.
        min_retweets (int): Minimum total number of retweets a user must have made to be
            considered active and included in the similarity computation. IDF is still
            computed over all users regardless of this threshold. Default is 2.
        min_overlap (int): Minimum number of distinct retweeted accounts that two users
            must share for an edge to be included in the graph. Default is 3.

    Returns:
        G (nx.Graph): Undirected weighted graph of active users. Edge weights are
            TF-IDF cosine similarity scores (0, 1], only present when the overlap
            condition is satisfied. Isolated nodes are excluded.
    
    This is a modified version of code used in the following paper:
    
    Luca Luceri, Valeria Pantè, Keith Burghardt, and Emilio Ferrara. 2024. 
    Unmasking the Web of Deceit: Uncovering Coordinated Activity to Expose Information Operations on Twitter. 
    In Proceedings of the ACM Web Conference 2024 (WWW '24). Association for Computing Machinery, New York, NY, USA, 2530–2541. 
    https://doi.org/10.1145/3589334.3645529
    """

    data = data.copy()
    
    data = data.rename(columns={userid_col: 'userid', feature_col: 'feature_shared', type_column: 'row_type'})

    data = data[data['row_type']=='retweet'] #keep only retweets

    temp = data.groupby('feature_shared', as_index=False).count()
    data = data.loc[data['feature_shared'].isin(temp.loc[temp['userid']>1]['feature_shared'].to_list())] #keep only accounts retweeted by more than 1 user

    # Count how many times each user retweeted each account (instead of binary 1)
    data = data.groupby(['userid', 'feature_shared'], as_index=False).size().rename(columns={'size': 'value'})

    # Identify active users (>min_retweets total retweets) BEFORE filtering, so IDF is computed over all users
    user_totals = data.groupby('userid')['value'].sum()
    active_users = set(user_totals[user_totals > min_retweets].index.astype(str))

    ids = dict(zip(list(data.feature_shared.unique()), list(range(data.feature_shared.unique().shape[0]))))
    data['feature_shared'] = data['feature_shared'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(data.userid.astype(str).unique()), list(range(data.userid.unique().shape[0]))))
    data['userid'] = data['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = CategoricalDtype(sorted(data.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.feature_shared.unique()), ordered=True)
    
    row = data.userid.astype(person_c).cat.codes
    col = data.feature_shared.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((data["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    # Fit TF-IDF on ALL users so IDF reflects account popularity across the full population
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)

    # Now filter to active users only (>2 retweets) for similarity computation
    userid_inv = {v: k for k, v in userid.items()}  # int index -> username
    active_indices = sorted([userid[u] for u in active_users if u in userid])
    active_usernames = [userid_inv[i] for i in active_indices]

    if not active_indices:
        return nx.Graph()

    tfidf_active = tfidf_matrix[active_indices, :]

    # --- Minimum overlap filter ---
    # Build a binary matrix (1 if user retweeted account at least once) for active users
    binary_active = (sparse_matrix[active_indices, :] > 0).astype(np.float32)
    # overlap[i, j] = number of accounts retweeted by both user i and user j
    overlap = (binary_active @ binary_active.T).toarray()

    similarities = cosine_similarity(tfidf_active, dense_output=False)

    # Apply hard overlap threshold: zero out pairs sharing fewer than min_overlap accounts
    overlap_mask = (overlap >= min_overlap).astype(np.float32)
    np.fill_diagonal(overlap_mask, 0)  # remove self-loops
    similarities = csr_matrix(similarities.toarray() * overlap_mask)

    df_adj = pd.DataFrame(similarities.toarray())


    df_adj.index = active_usernames
    df_adj.columns = active_usernames
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


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
    keep_na: bool = False,                                  # drop None/NaN extras by default
    return_sigma = True
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
    return sigma_viz if return_sigma else graph