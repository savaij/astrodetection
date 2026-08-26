import re
import warnings
from typing import Dict, Iterable, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlsplit

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

_URL_RE = re.compile(r'https?://[^\s<>"\'()\[\]{}]+', re.IGNORECASE)

# Trailing characters the regex greedily captures when a URL ends a sentence.
_URL_TRAILING_PUNCT = '.,;:!?)]}\'"'

# Query parameters dropped during normalization: they identify the campaign or the
# sharer, not the resource, so keeping them would split the same page into many
# distinct features and break the co-sharing match between users.
_TRACKING_PARAMS = frozenset({
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'utm_id',
    'fbclid', 'gclid', 'gclsrc', 'dclid', 'msclkid', 'yclid', 'twclid',
    'mc_cid', 'mc_eid', 'igshid', 'ref_src', 'ref_url', 's', 't', 'si',
})


def create_coSharing_graph(data, type_column='row_type', userid_col='screen name', feature_col='retweeted user', min_retweets=3, min_overlap=3, fast_graph=False, weight_threshold=0.9):
    """
    Build a co-sharing similarity graph among users based on shared retweet targets.

    Each node represents an active user (at least `min_retweets` total retweets). An edge between
    two users is added when they share at least `min_overlap` retweeted accounts in
    common, weighted by the TF-IDF cosine similarity of their retweet vectors.

    Algorithm steps:
        1. Filter to retweet rows only and keep accounts retweeted by more than one user.
        2. Count how many times each user retweeted each account (frequency matrix).
        3. Compute TF-IDF weights over the full user population so that IDF reflects
           the global popularity of each retweeted account.
        4. Restrict similarity computation to active users (>= `min_retweets` total retweets).
        5. Compute pairwise cosine similarity among active users' TF-IDF vectors.
        6. Apply a hard overlap filter: retain only pairs sharing at least
           `min_overlap` distinct retweeted accounts.
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
            computed over all users regardless of this threshold. Default is 3.
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


def _keep_shared_features(data):
    """
    Keep only the rows whose `feature_shared` value appears in more than one row.

    NB: this counts ROWS, not distinct users (a feature used 5 times by a single user
    passes). Changing the filter would change the IDF and thus all similarities, so the
    behaviour is preserved exactly as it was when duplicated inside the two builders.

    The operation is idempotent: it drops whole features, which leaves the row counts of
    the surviving features untouched. Callers may therefore apply it before handing the
    table to a builder that applies it again.

    Args:
        data (pd.DataFrame): Long-format table with at least the columns 'userid' and
            'feature_shared'. Extra columns are ignored and carried through.

    Returns:
        pd.DataFrame: The subset of rows whose feature is shared.
    """

    feat_rows = data.groupby('feature_shared')['userid'].count()
    return data.loc[data['feature_shared'].isin(feat_rows.index[feat_rows > 1])]


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
        min_overlap (int): Minimum number of distinct feature values that two users
            must share for an edge to be included.

    Returns:
        nx.Graph: Undirected weighted graph of active users. Edge weights are
            TF-IDF cosine similarity scores. Isolated nodes are excluded.
    """

    data = data.copy()

    data = _keep_shared_features(data) #keep only features used in more than 1 row

    # Nothing shared: bail out before TfidfTransformer, which rejects a (0, 0) matrix.
    # Mirrors the early return in _tfidf_cosine_overlap_graph_fast.
    if data.empty:
        return nx.Graph()

    # Count how many times each user produced each feature (instead of binary 1)
    data = data.groupby(['userid', 'feature_shared'], as_index=False).size().rename(columns={'size': 'value'})

    # Identify active users (>=min_count total events) BEFORE filtering, so IDF is computed over all users
    user_totals = data.groupby('userid')['value'].sum()
    active_users = set(user_totals[user_totals >= min_count].index.astype(str))

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

    # Now filter to active users only (>=min_count events) for similarity computation
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

    # Apply hard overlap threshold: retain only pairs sharing at least min_overlap features
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

    data = _keep_shared_features(data)
    if data.empty:
        G = nx.Graph()
        G.graph['weight_threshold'] = weight_threshold
        return G

    # Count how many times each user produced each feature (instead of binary 1)
    data = data.groupby(['userid', 'feature_shared'], as_index=False).size().rename(columns={'size': 'value'})

    # Identify active users (>=min_count total events) BEFORE filtering, so IDF is computed over all users
    user_totals = data.groupby('userid')['value'].sum()
    active_users = set(user_totals[user_totals >= min_count].index.astype(str))

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


def create_coActivity_graph(data, userid_col='screen name', timestamp_col='tweet_date', bin_minutes=5, min_activity=3, min_overlap=3, fast_graph=False, weight_threshold=0.9):
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
        5. Restrict similarity computation to active users (>= `min_activity` total events).
        6. Compute pairwise cosine similarity and apply a hard overlap filter, retaining only
           pairs sharing at least `min_overlap` distinct time bins.
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
            users regardless of this threshold. Default is 3.
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


def _extract_urls(text):
    """
    Extract http/https URLs from a free-text string.

    Args:
        text: Any value; non-string values yield an empty list.

    Returns:
        list[str]: Matched URLs, stripped of the sentence punctuation the regex
            greedily captures when a link ends a phrase.
    """
    if not isinstance(text, str) or not text:
        return []

    return [u.rstrip(_URL_TRAILING_PUNCT) for u in _URL_RE.findall(text)]


def _normalize_url(url, granularity='url', strip_tracking_params=True):
    """
    Reduce a URL to a canonical key so that cosmetic variants of the same link map
    to the same feature.

    Normalization: scheme dropped (http and https collapse), host lowercased without
    a leading 'www.' or a default port, fragment dropped, tracking parameters removed,
    remaining query parameters sorted, trailing slash removed.

    Args:
        url (str): The raw URL.
        granularity (str): 'url' => host + path + query; 'domain' => host only.
        strip_tracking_params (bool): Whether to drop `_TRACKING_PARAMS` from the query.

    Returns:
        str | None: The canonical key, or None when the URL has no host.
    """
    if granularity not in ('url', 'domain'):
        raise ValueError("granularity must be either 'url' or 'domain'")

    if not isinstance(url, str) or not url:
        return None

    try:
        parts = urlsplit(url if '://' in url else f'http://{url}')
    except ValueError:
        return None

    host = parts.hostname  # already lowercased, port and credentials removed
    # The scheme-less fallback above makes urlsplit accept almost anything as a host,
    # so require something that at least looks like a hostname.
    if not host or '.' not in host or any(c.isspace() for c in host):
        return None
    if host.startswith('www.'):
        host = host[4:]

    if granularity == 'domain':
        return host

    path = parts.path.rstrip('/')

    query = ''
    if parts.query:
        params = parse_qsl(parts.query, keep_blank_values=True)
        if strip_tracking_params:
            params = [(k, v) for k, v in params if k.lower() not in _TRACKING_PARAMS]
        if params:
            query = '?' + urlencode(sorted(params))

    return f'{host}{path}{query}'


def create_coURL_graph(data, userid_col='screen name', text_col='tweet', url_col=None, type_column=None, include_types=None, url_granularity='url', strip_tracking_params=True, exclude_domains=None, min_urls=3, min_url_posts=2, min_overlap=3, fast_graph=False, weight_threshold=0.9):
    """
    Build a co-URL similarity graph among users based on shared links.

    Third sibling of create_coSharing_graph (feature = retweeted account) and
    create_coActivity_graph (feature = time bin): here the shared feature is the
    normalized URL a user posted. Accounts that repeatedly push the same external
    links -- especially links few other accounts share -- get a high TF-IDF cosine
    similarity, a common signature of coordinated amplification campaigns.

    Algorithm steps:
        1. Optionally restrict rows by type (`type_column` / `include_types`).
        2. Collect the URLs of each row: from `url_col` when given, otherwise by
           regex over `text_col`. Explode to one (user, URL) pair per row.
        3. Normalize each URL to a canonical key (see `_normalize_url`) and drop
           duplicates within the same source row.
        4. Keep only URLs used in more than one row.
        5. Count how many times each user shared each URL (frequency matrix).
        6. Compute TF-IDF weights over the full user population, so that IDF
           down-weights links everybody shares and up-weights rare ones.
        7. Restrict similarity computation to active users. Both conditions must hold:
           at least `min_urls` URL events AND at least `min_url_posts` distinct posts
           carrying one of those URLs.
        8. Compute pairwise cosine similarity and apply a hard overlap filter,
           retaining only pairs sharing at least `min_overlap` distinct URLs.
        9. Build an undirected weighted graph; remove self-loops and isolated nodes.

    Args:
        data (pd.DataFrame): DataFrame containing post/retweet rows. Must include a
            user id column and either `text_col` or `url_col`.
        userid_col (str): Name of the column containing user identifiers. Default is
            'screen name'.
        text_col (str): Name of the column containing the post text URLs are extracted
            from. Ignored when `url_col` is given. Default is 'tweet'.
        url_col (str, optional): Name of a column that already holds the URLs. Takes
            precedence over `text_col`. Cells may be a list/tuple/set/array of URLs or
            a string (parsed with the same regex, so several URLs per cell are fine).
            Prefer this when the column holds *expanded* URLs. Default is None.
        type_column (str, optional): Name of the column identifying the row type. Only
            used together with `include_types`. Default is None (all rows).
        include_types (iterable, optional): Row types to keep, e.g. ('post',) to drop
            retweets. Retweets inherit the links of the original post, so including
            them makes this signal partly redundant with create_coSharing_graph.
            Default is None (all rows).
        url_granularity (str): 'url' to match on the full normalized link, 'domain' to
            match on the host only. Domain level is a much coarser (and noisier) signal
            since legitimate accounts also share the same news sites. Default is 'url'.
        strip_tracking_params (bool): Whether to drop tracking query parameters
            (utm_*, fbclid, ...) during normalization. Default is True.
        exclude_domains (iterable, optional): Hosts to ignore (matched after
            normalization, so without 'www.'). Pass {'twitter.com', 'x.com'} for a
            signal independent of co-retweeting. Default is None.
        min_urls (int): Minimum total number of URLs a user must have shared to be
            considered active and included in the similarity computation. IDF is still
            computed over all users regardless of this threshold. Default is 3.
        min_url_posts (int): Minimum number of distinct posts (one row = one post) that
            must carry at least one of the user's shared URLs. Cumulative with
            `min_urls`: a single post packed with links no longer makes an account
            eligible, since it carries no evidence of repetition over time. Like
            `min_urls`, this is counted on the URLs that survive step 4, and IDF is
            still computed over all users. With `min_urls=2, min_url_posts=2` this
            accepts `post 1: URL A` + `post 2: URL A` (two events, two posts) and
            rejects `post 1: URL A + URL B` (two events, one post) as well as
            `post 1: URL A + URL A` (deduplicated: one event, one post). Pass 1 to
            reproduce the eligibility criterion of versions < 0.3.0. Default is 2.
        min_overlap (int): Minimum number of distinct URLs that two users must share for
            an edge to be included in the graph. Default is 3, for consistency with the
            sibling builders; URL sharing is sparser than retweeting, so this is the first
            parameter to lower when the graph comes out empty.
        fast_graph (bool): If True, use the memory-bounded builder that applies the
            weight threshold during construction (see `weight_threshold`) instead of
            materializing dense N x N matrices. The resulting graph is NOT the complete
            graph: edges with weight < `weight_threshold` are never created. Use this on
            large datasets where the full graph exhausts RAM. Default is False.
        weight_threshold (float): Only used when `fast_graph=True`. Minimum edge weight
            (TF-IDF cosine similarity) to keep during construction. IMPORTANT: the fast
            graph is equivalent to the full graph only if the threshold later passed to
            get_similarity_hub_score / compute_bot_likelihood_metrics (`url_threshold`)
            is >= this value; a lower downstream threshold yields a silently
            under-connected graph. Default is 0.9.

    Returns:
        nx.Graph: Undirected weighted graph of active users. Edge weights are TF-IDF
            cosine similarity scores. Isolated nodes are excluded.

    Note:
        t.co short links are generally unique per tweet, so the same destination appears
        as several distinct URLs and the graph comes out empty or misleading. A warning
        is emitted when most extracted links are t.co: in that case pass `url_col` with
        the expanded URLs.
    """

    if not isinstance(min_url_posts, (int, np.integer)) or min_url_posts < 1:
        raise ValueError("min_url_posts must be an integer >= 1")

    data = data.copy()

    data = data.rename(columns={userid_col: 'userid'})

    if type_column is not None and include_types is not None:
        data = data[data[type_column].isin(list(include_types))]

    if url_col is not None:
        def _row_urls(value):
            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                return [u for u in value if isinstance(u, str) and u]
            return _extract_urls(value)

        urls = data[url_col].map(_row_urls)
    else:
        urls = data[text_col].map(_extract_urls)

    # Row id assigned before exploding, so the same link repeated inside one post can be
    # deduplicated without collapsing repetitions across posts (which do feed the TF).
    data = data[['userid']].copy()
    data['_row_id'] = np.arange(len(data))
    data['feature_shared'] = urls.values

    data = data.explode('feature_shared').dropna(subset=['feature_shared'])

    if not data.empty:
        # Normalize the unique raw URLs only: the same link recurs many times and
        # parsing is the expensive part here.
        raw_uniques = data['feature_shared'].unique()
        norm_map = {u: _normalize_url(u, granularity=url_granularity, strip_tracking_params=strip_tracking_params) for u in raw_uniques}
        data['feature_shared'] = data['feature_shared'].map(norm_map)
        data = data.dropna(subset=['feature_shared'])

    if not data.empty:
        # The normalized key always starts with the host, which contains neither '/' nor '?'.
        hosts = data['feature_shared'].str.split('/', n=1).str[0].str.split('?', n=1).str[0]

        tco_share = (hosts == 't.co').mean()
        if tco_share > 0.5:
            warnings.warn(
                f"{tco_share:.0%} of the extracted links are t.co short links. These are generally "
                "unique per tweet, so identical destinations look like distinct URLs and the "
                "resulting graph will be empty or misleading. Pass url_col with expanded URLs.",
                stacklevel=2,
            )

        if exclude_domains:
            # positional mask: the index carries duplicates after the explode
            data = data[(~hosts.isin(set(exclude_domains))).to_numpy()]

        data = data.drop_duplicates(subset=['_row_id', 'feature_shared'])

    data = data[['userid', 'feature_shared', '_row_id']]

    # Guard: the shared builders reach TfidfTransformer.fit_transform on a (0, 0)
    # matrix before their own early return.
    if data.empty:
        G = nx.Graph()
        if fast_graph:
            G.graph['weight_threshold'] = weight_threshold
        return G

    # Distinct posts counted on the URLs that survive the shared-feature filter, for
    # consistency with min_urls (which the builders apply after that same filter).
    eligible_users = None
    if min_url_posts > 1:
        n_posts = _keep_shared_features(data).groupby('userid')['_row_id'].nunique()
        eligible_users = set(n_posts.index[n_posts >= min_url_posts].astype(str))

    events = data[['userid', 'feature_shared']]

    if fast_graph:
        G = _tfidf_cosine_overlap_graph_fast(events, min_count=min_urls, min_overlap=min_overlap, weight_threshold=weight_threshold)
    else:
        G = _tfidf_cosine_overlap_graph(events, min_count=min_urls, min_overlap=min_overlap)

    if eligible_users is not None:
        # Equivalent to filtering inside the builder, without adding a parameter to it:
        # IDF has already been fitted over the whole population, and both the cosine
        # weight and the overlap count are pairwise, so they do not depend on which rows
        # were selected. Every edge touching an ineligible user disappears with its node.
        # The second isolates pass catches users whose only edges pointed at one.
        G.remove_nodes_from([n for n in list(G) if str(n) not in eligible_users])
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
