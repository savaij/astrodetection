"""
Rimpiazzo memory-bounded di astrodetection.network_utilities._tfidf_cosine_overlap_graph.

Perche' esiste: la versione della libreria materializza quattro matrici DENSE
N x N (N = utenti attivi) -- overlap, similarities.toarray(), overlap_mask e
df_adj. Con N ~ 37k (gruppo control di China_1) sono ~10 GB ciascuna: OOM
garantito su qualunque macchina ragionevole.

Perche' e' lecito: il grafo completo viene poi passato a get_similarity_hub_score,
che come prima cosa scarta tutti gli archi con weight < threshold (0.9). Gli archi
sotto soglia non influenzano mai il risultato, quindi li si puo' non generare
affatto invece di generarli e poi buttarli.

Equivalenza garantita (verificata da validate_fast_graphs.py contro la libreria):
- stessa matrice TF-IDF, stesso IDF calcolato su TUTTI gli utenti;
- stesso filtro min_overlap sul numero di feature distinte condivise;
- stessi nodi, aggiunti nello stesso ordine (conta per il tie-breaking di
  greedy_modularity_communities), stessi archi, stessi pesi;
- l'unica differenza e' che gli archi con weight < weight_threshold non vengono
  mai creati -- esattamente quelli che il chiamante rimuove subito dopo.

Se astrodetection cambia _tfidf_cosine_overlap_graph, questo file va risincronizzato.
"""

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

# Righe di utenti per blocco nel prodotto di similarita'. Il picco di memoria e'
# ~ BLOCK x N valori: 512 x 40k x 12 byte ~ 250 MB nel caso peggiore (blocco denso).
BLOCK_ROWS = 512


def tfidf_cosine_overlap_graph(data, min_count, min_overlap, weight_threshold):
    """Come _tfidf_cosine_overlap_graph, ma genera solo gli archi con
    weight >= weight_threshold e non alloca mai una matrice densa N x N.

    data: DataFrame long con colonne 'userid' e 'feature_shared'.
    """
    data = data[["userid", "feature_shared"]]

    # 1) tieni solo le feature con piu' di una riga.
    # NB: la libreria qui usa groupby(...).count(), che conta le RIGHE, non gli
    # utenti distinti, nonostante il suo commento dica "shared by more than 1
    # user" -- una feature usata 5 volte da un solo utente passa il filtro.
    # Replichiamo il codice, non il commento: cambiare il filtro cambierebbe
    # l'IDF e quindi tutte le similarita'.
    feat_rows = data.groupby("feature_shared")["userid"].count()
    keep = feat_rows.index[feat_rows > 1]
    data = data.loc[data["feature_shared"].isin(keep)]
    if data.empty:
        return nx.Graph()

    # 2) matrice di conteggio utente x feature
    data = data.groupby(["userid", "feature_shared"], as_index=False).size().rename(
        columns={"size": "value"}
    )

    # 3) utenti attivi (> min_count eventi totali), IDF resta su tutta la popolazione
    user_totals = data.groupby("userid")["value"].sum()
    active_users = set(user_totals[user_totals > min_count].index.astype(str))

    # Stessa codifica della libreria: interi assegnati nell'ordine di prima
    # comparsa, che determina l'ordine dei nodi nel grafo finale.
    feat_uniques = data["feature_shared"].unique()
    ids = dict(zip(list(feat_uniques), range(len(feat_uniques))))
    data["feature_shared"] = data["feature_shared"].map(ids).astype(int)
    del ids

    user_uniques = data["userid"].astype(str).unique()
    userid = dict(zip(list(user_uniques), range(len(user_uniques))))
    data["userid"] = data["userid"].astype(str).map(userid).astype(int)

    person_c = CategoricalDtype(sorted(data.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(data.feature_shared.unique()), ordered=True)
    row = data.userid.astype(person_c).cat.codes
    col = data.feature_shared.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix(
        (data["value"], (row, col)),
        shape=(person_c.categories.size, thing_c.categories.size),
    )
    del row, col, person_c, thing_c, data

    tfidf_matrix = TfidfTransformer().fit_transform(sparse_matrix)

    userid_inv = {v: k for k, v in userid.items()}
    active_indices = sorted(userid[u] for u in active_users if u in userid)
    if not active_indices:
        return nx.Graph()
    active_usernames = [userid_inv[i] for i in active_indices]
    del userid, userid_inv

    tfidf_active = tfidf_matrix[active_indices, :]
    del tfidf_matrix
    binary_active = (sparse_matrix[active_indices, :] > 0).astype(np.float32)
    del sparse_matrix

    # cosine_similarity(X) == (X_normalizzata) @ (X_normalizzata).T; normalizzando
    # una volta sola, ogni blocco e' un semplice prodotto sparso.
    tfidf_norm = normalize(tfidf_active, norm="l2", axis=1, copy=True)
    del tfidf_active

    n_active = len(active_indices)
    src, dst, wts = [], [], []
    for start in range(0, n_active, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n_active)
        sims = tfidf_norm[start:stop] @ tfidf_norm.T  # (blocco x N), sparso

        # Soglia subito: sopra 0.9 sopravvive una frazione minima delle coppie.
        sims.data[sims.data < weight_threshold] = 0
        sims.eliminate_zeros()
        if sims.nnz == 0:
            continue

        coo = sims.tocoo()
        gi = coo.row + start
        gj = coo.col
        w = coo.data
        del sims, coo

        # solo il triangolo superiore: niente self-loop, niente doppioni
        upper = gi < gj
        if not upper.any():
            continue
        gi, gj, w = gi[upper], gj[upper], w[upper]

        # overlap solo per le coppie sopravvissute (poche): niente matrice densa
        ov = np.asarray(
            binary_active[gi].multiply(binary_active[gj]).sum(axis=1)
        ).ravel()
        ok = ov >= min_overlap
        if not ok.any():
            continue

        src.append(gi[ok])
        dst.append(gj[ok])
        wts.append(w[ok])

    # Stessi nodi e stesso ordine di nx.from_pandas_adjacency(df_adj): tutti gli
    # utenti attivi, poi gli isolati vengono rimossi.
    G = nx.Graph()
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


def create_coSharing_graph_fast(data, type_column, userid_col, feature_col,
                                weight_threshold, min_retweets=2, min_overlap=3):
    data = data.rename(
        columns={userid_col: "userid", feature_col: "feature_shared", type_column: "row_type"}
    )
    data = data[data["row_type"] == "retweet"]
    if data.empty:
        return nx.Graph()
    return tfidf_cosine_overlap_graph(
        data[["userid", "feature_shared"]],
        min_count=min_retweets,
        min_overlap=min_overlap,
        weight_threshold=weight_threshold,
    )


def create_coActivity_graph_fast(data, userid_col, timestamp_col, weight_threshold,
                                 bin_minutes=5, min_activity=2, min_overlap=3):
    data = data.rename(columns={userid_col: "userid"})
    ts = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce").dt.tz_localize(None)
    data = data.loc[ts.notna()].copy()
    ts = ts.loc[ts.notna()]
    data["feature_shared"] = (
        ts.dt.floor(f"{bin_minutes}min") - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta(seconds=1)
    del ts
    if data.empty:
        return nx.Graph()
    return tfidf_cosine_overlap_graph(
        data[["userid", "feature_shared"]],
        min_count=min_activity,
        min_overlap=min_overlap,
        weight_threshold=weight_threshold,
    )
