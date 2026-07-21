from typing import Dict, Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from ipysigma import Sigma


# Rows of active users processed per block in the fast similarity product. Peak
# memory is ~ _BLOCK_ROWS x N values, keeping the fast path memory-bounded.
_BLOCK_ROWS = 512


def create_coSharing_graph(data, type_column='row_type', userid_col='screen name', feature_col='retweeted user', min_retweets=2, min_overlap=3, fast_graph=False, weight_threshold=0.9):
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
        fast_graph (bool): If True, use the memory-bounded builder that applies the
            weight threshold during construction (see `weight_threshold`) instead of
            materializing dense N x N matrices. The resulting graph is NOT the complete
            graph: edges with weight < `weight_threshold` are never created. Use this on
            large datasets where the full graph exhausts RAM. Default is False.
        weight_threshold (float): Only used when `fast_graph=True`. Minimum edge weight
            (TF-IDF cosine similarity) to keep during construction. IMPORTANT: the fast
            graph is equivalent to the full graph only if the threshold later passed to
            get_similarity_hub_score / compute_bot_likelihood_metrics
            (`similarity_sharing_threshold`) is >= this value; a lower downstream
            threshold yields a silently under-connected graph. Default is 0.9.

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

    if fast_graph:
        return _tfidf_cosine_overlap_graph_fast(data[['userid', 'feature_shared']], min_count=min_retweets, min_overlap=min_overlap, weight_threshold=weight_threshold)

    return _tfidf_cosine_overlap_graph(data[['userid', 'feature_shared']], min_count=min_retweets, min_overlap=min_overlap)


def _tfidf_cosine_overlap_graph(data, min_count, min_overlap):
    """
    Build a TF-IDF cosine-similarity graph among users from a long-format
    (userid, feature_shared) event table.

    Shared core of create_coSharing_graph and create_coActivity_graph: the
    `feature_shared` column is treated as an opaque categorical key (a retweeted
    account, a time bin, ...). See those functions for the full algorithm.

    Args:
        data (pd.DataFrame): One row per event, with columns 'userid' (str) and
            'feature_shared' (hashable). One column per distinct feature value.
        min_count (int): Minimum total events a user must have to be considered
            active and included in the similarity computation. IDF is still
            computed over all users regardless of this threshold.
        min_overlap (int): Minimum number of distinct feature values that two
            users must share for an edge to be included.

    Returns:
        nx.Graph: Undirected weighted graph of active users. Edge weights are
            TF-IDF cosine similarity scores. Isolated nodes are excluded.
    """

    data = data.copy()

    temp = data.groupby('feature_shared', as_index=False).count()
    data = data.loc[data['feature_shared'].isin(temp.loc[temp['userid']>1]['feature_shared'].to_list())] #keep only features shared by more than 1 user

    # Count how many times each user produced each feature (instead of binary 1)
    data = data.groupby(['userid', 'feature_shared'], as_index=False).size().rename(columns={'size': 'value'})

    # Identify active users (>min_count total events) BEFORE filtering, so IDF is computed over all users
    user_totals = data.groupby('userid')['value'].sum()
    active_users = set(user_totals[user_totals > min_count].index.astype(str))

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
    
    # Fit TF-IDF on ALL users so IDF reflects feature popularity across the full population
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)

    # Now filter to active users only (>min_count events) for similarity computation
    userid_inv = {v: k for k, v in userid.items()}  # int index -> username
    active_indices = sorted([userid[u] for u in active_users if u in userid])
    active_usernames = [userid_inv[i] for i in active_indices]

    if not active_indices:
        return nx.Graph()

    tfidf_active = tfidf_matrix[active_indices, :]

    # --- Minimum overlap filter ---
    # Build a binary matrix (1 if user produced the feature at least once) for active users
    binary_active = (sparse_matrix[active_indices, :] > 0).astype(np.float32)
    # overlap[i, j] = number of features shared by both user i and user j
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


def _tfidf_cosine_overlap_graph_fast(data, min_count, min_overlap, weight_threshold):
    """
    Memory-bounded equivalent of _tfidf_cosine_overlap_graph: builds only the edges
    with weight >= `weight_threshold` and never allocates a dense N x N matrix.

    Same TF-IDF (IDF fitted over ALL users), same min_overlap filter, same nodes in
    the same order, same edges and weights as _tfidf_cosine_overlap_graph -- the only
    difference is that edges with weight < `weight_threshold` are never created. This
    is safe because get_similarity_hub_score discards exactly those edges immediately
    afterwards, as long as its threshold is >= `weight_threshold`.

    The chosen `weight_threshold` is stored on the returned graph as
    G.graph['weight_threshold'] so get_similarity_hub_score can warn when it is asked
    to filter at a lower (invalid) threshold.

    Args:
        data (pd.DataFrame): One row per event, with columns 'userid' (str) and
            'feature_shared' (hashable).
        min_count (int): Minimum total events a user must have to be considered active.
            IDF is still computed over all users regardless of this threshold.
        min_overlap (int): Minimum number of distinct feature values that two users
            must share for an edge to be included.
        weight_threshold (float): Minimum edge weight (cosine similarity) to keep. Edges
            below this value are never generated.

    Returns:
        nx.Graph: Undirected weighted graph of active users, carrying
            G.graph['weight_threshold']. Isolated nodes are excluded.
    """

    data = data[['userid', 'feature_shared']]

    # Keep only features with more than one row.
    # NB: like _tfidf_cosine_overlap_graph, this counts ROWS, not distinct users
    # (a feature used 5 times by a single user passes). We replicate the code, not
    # the comment: changing the filter would change the IDF and thus all similarities.
    feat_rows = data.groupby('feature_shared')['userid'].count()
    keep = feat_rows.index[feat_rows > 1]
    data = data.loc[data['feature_shared'].isin(keep)]
    if data.empty:
        G = nx.Graph()
        G.graph['weight_threshold'] = weight_threshold
        return G

    # Count how many times each user produced each feature (instead of binary 1)
    data = data.groupby(['userid', 'feature_shared'], as_index=False).size().rename(columns={'size': 'value'})

    # Identify active users (>min_count total events) BEFORE filtering, so IDF is computed over all users
    user_totals = data.groupby('userid')['value'].sum()
    active_users = set(user_totals[user_totals > min_count].index.astype(str))

    # Same encoding as _tfidf_cosine_overlap_graph: integers assigned in order of
    # first appearance, which determines node order in the final graph.
    feat_uniques = data['feature_shared'].unique()
    ids = dict(zip(list(feat_uniques), range(len(feat_uniques))))
    data['feature_shared'] = data['feature_shared'].map(ids).astype(int)
    del ids

    user_uniques = data['userid'].astype(str).unique()
    userid = dict(zip(list(user_uniques), range(len(user_uniques))))
    data['userid'] = data['userid'].astype(str).map(userid).astype(int)

    person_c = CategoricalDtype(sorted(data.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.feature_shared.unique()), ordered=True)
    row = data.userid.astype(person_c).cat.codes
    col = data.feature_shared.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((data["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c, data

    # Fit TF-IDF on ALL users so IDF reflects feature popularity across the full population
    tfidf_matrix = TfidfTransformer().fit_transform(sparse_matrix)

    userid_inv = {v: k for k, v in userid.items()}
    active_indices = sorted(userid[u] for u in active_users if u in userid)
    if not active_indices:
        G = nx.Graph()
        G.graph['weight_threshold'] = weight_threshold
        return G
    active_usernames = [userid_inv[i] for i in active_indices]
    del userid, userid_inv

    tfidf_active = tfidf_matrix[active_indices, :]
    del tfidf_matrix
    binary_active = (sparse_matrix[active_indices, :] > 0).astype(np.float32)
    del sparse_matrix

    # cosine_similarity(X) == (X_normalized) @ (X_normalized).T; normalizing once,
    # each block is a plain sparse product.
    tfidf_norm = normalize(tfidf_active, norm="l2", axis=1, copy=True)
    del tfidf_active

    n_active = len(active_indices)
    src, dst, wts = [], [], []
    for start in range(0, n_active, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n_active)
        sims = tfidf_norm[start:stop] @ tfidf_norm.T  # (block x N), sparse

        # Threshold immediately: above the cutoff only a tiny fraction of pairs survive.
        sims.data[sims.data < weight_threshold] = 0
        sims.eliminate_zeros()
        if sims.nnz == 0:
            continue

        coo = sims.tocoo()
        gi = coo.row + start
        gj = coo.col
        w = coo.data
        del sims, coo

        # upper triangle only: no self-loops, no duplicates
        upper = gi < gj
        if not upper.any():
            continue
        gi, gj, w = gi[upper], gj[upper], w[upper]

        # overlap only for the surviving (few) pairs: no dense matrix
        ov = np.asarray(binary_active[gi].multiply(binary_active[gj]).sum(axis=1)).ravel()
        ok = ov >= min_overlap
        if not ok.any():
            continue

        src.append(gi[ok])
        dst.append(gj[ok])
        wts.append(w[ok])

    # Same nodes and same order as nx.from_pandas_adjacency(df_adj): all active users,
    # then isolates are removed.
    G = nx.Graph()
    G.graph['weight_threshold'] = weight_threshold
    G.add_nodes_from(active_usernames)
    if src:
        src = np.concatenate(src)
        dst = np.concatenate(dst)
        wts = np.concatenate(wts)
        G.add_edges_from(
            (active_usernames[i], active_usernames[j], {"weight": float(w)})
            for i, j, w in zip(src, dst, wts)
        )

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def create_coActivity_graph(data, userid_col='screen name', timestamp_col='tweet_date', bin_minutes=5, min_activity=2, min_overlap=3, fast_graph=False, weight_threshold=0.9):
    """
    Build a co-activity similarity graph among users based on shared temporal activity bins.

    Temporal counterpart of create_coSharing_graph: instead of the retweeted account, the
    shared feature is the time bin in which a user was active. Posts and retweets are treated
    the same (no row-type filter). Each user's activity timestamps are floored to `bin_minutes`
    buckets; users that are repeatedly active in the same (and especially rare) time bins get a
    high TF-IDF cosine similarity, which is a signal of coordinated behaviour.

    Algorithm steps:
        1. Parse `timestamp_col` (timezone-normalized), drop unparseable rows, and floor each
           activity time to a `bin_minutes` bucket -> the shared feature is the unix timestamp
           (seconds) of the bin.
        2. Keep only bins used by more than one user.
        3. Count how many times each user was active in each bin (frequency matrix).
        4. Compute TF-IDF weights over the full user population so that IDF down-weights common
           bins (everyone active) and up-weights rare bins shared by few users.
        5. Restrict similarity computation to active users (> `min_activity` total events).
        6. Compute pairwise cosine similarity and apply a hard overlap filter (zero out pairs
           sharing fewer than `min_overlap` distinct time bins).
        7. Build an undirected weighted graph; remove self-loops and isolated nodes.

    Args:
        data (pd.DataFrame): DataFrame containing activity rows (posts and/or retweets). Must
            include columns for user id and activity timestamp.
        userid_col (str): Name of the column containing user identifiers. Default is 'screen name'.
        timestamp_col (str): Name of the column containing the activity timestamp. Default is
            'tweet_date'.
        bin_minutes (int): Size of the temporal bin in minutes. Default is 5.
        min_activity (int): Minimum total number of events a user must have to be considered
            active and included in the similarity computation. IDF is still computed over all
            users regardless of this threshold. Default is 2.
        min_overlap (int): Minimum number of distinct time bins that two users must share for an
            edge to be included in the graph. Default is 3.
        fast_graph (bool): If True, use the memory-bounded builder that applies the weight
            threshold during construction (see `weight_threshold`) instead of materializing
            dense N x N matrices. The resulting graph is NOT the complete graph: edges with
            weight < `weight_threshold` are never created. Use this on large datasets where the
            full graph exhausts RAM. Default is False.
        weight_threshold (float): Only used when `fast_graph=True`. Minimum edge weight (TF-IDF
            cosine similarity) to keep during construction. IMPORTANT: the fast graph is
            equivalent to the full graph only if the threshold later passed to
            get_similarity_hub_score / compute_bot_likelihood_metrics (`temporal_threshold`) is
            >= this value; a lower downstream threshold yields a silently under-connected graph.
            Default is 0.9.

    Returns:
        nx.Graph: Undirected weighted graph of active users. Edge weights are TF-IDF cosine
            similarity scores. Isolated nodes are excluded.

    Note:
        `feature_shared` is an opaque categorical key mapped to a matrix column; the TF-IDF is
        computed by TfidfTransformer over a hand-built count matrix (no text tokenization), so
        the bin encoding (unix int here) is purely a representation choice.
    """

    data = data.copy()

    data = data.rename(columns={userid_col: 'userid'})

    ts = pd.to_datetime(data[timestamp_col], utc=True, errors='coerce').dt.tz_localize(None)
    data = data.loc[ts.notna()].copy()
    ts = ts.loc[ts.notna()]

    # Shared feature = unix timestamp (seconds) of the floored time bin.
    # Use Timedelta floor-division so the result is correct regardless of the
    # underlying datetime64 resolution (s/ms/us/ns).
    data['feature_shared'] = (ts.dt.floor(f'{bin_minutes}min') - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    if fast_graph:
        return _tfidf_cosine_overlap_graph_fast(data[['userid', 'feature_shared']], min_count=min_activity, min_overlap=min_overlap, weight_threshold=weight_threshold)

    return _tfidf_cosine_overlap_graph(data[['userid', 'feature_shared']], min_count=min_activity, min_overlap=min_overlap)


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