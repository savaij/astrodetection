from functools import wraps
import os
import re
import time
from typing import Union
import demoji
import faiss
#import fasttext
import networkx as nx
import numpy as np
import pandas as pd
from polyleven import levenshtein
import requests
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_text
import unicodedata
from tqdm.contrib.concurrent import thread_map
from tqdm.auto import trange
import networkx as nx


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        print(f">>> Start {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        if total_time < 60:
            print(f"<<< End {func.__name__}, Took: {total_time:.4f} sec")
        else:
            print(f"<<< End {func.__name__}, Took:{np.round((total_time)/60, 1)} min")
        return result

    return timeit_wrapper


def grouper(iterable, n):
    """A (lazy) iterator that chunks `iterable` into lists of `n`"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


###############################
#### Preprocessing Dataset ####
###############################


def deaccent(text):
    """
    Remove accentuation from the given string. Input must be a Unicode string.

    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    'Sef chomutovskych komunistu dostal postou bily prasek'
    """
    if not isinstance(text, str):
        text = text.decode('utf8')  # solo se accetti anche byte string
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def preprocess_text(
    s,
    lower=True,
    remove_accents=True,
    remove_urls=True,
    remove_mentions=True,
    remove_emojis=True,
    remove_hashtags_frontend=False,
    remove_twitter_cropend=False,
    replace_newline_characters=True,
    remove_punctuation=False,
):
    """
    clean a list-like of strings, performing all the following treatments by default
    Args:
        s (list-like of strings): input list-like of strings
        lower (bool, optional): lowercase the text. Defaults to True.
        remove_accents (bool, optional): deaccent the text. Defaults to True.
        remove_urls (bool, optional): remove urls from the text. Defaults to True.
        remove_mentions (bool, optional): remove mentions from the text. Defaults to True.
        remove_emojis (bool, optional): remove emojis from the text. Defaults to True.
        remove_hashtags_frontend (bool, optional): remove leading and ending hashtags from the text. Defaults to False.
        remove_twitter_cropend (bool, optional): remove Twitter-added "…" character at the end of messages that are too long. Defaults to False.
        replace_newline_characters (bool, optional): replace two commonly found escape characters: \r and \n with '. '. Defaults to True.
        remove_punctuation (bool, optional): remove punctuation from the text, be careful, it will remove # of hashtags too. Defaults to False.
    """
    if s is None:
        s = ""

    assert isinstance(s, (str, list, pd.Series, set, frozenset))

    if isinstance(s, str):
        encapsulated = True
        s = [s]
    else:
        encapsulated = False
    if lower:
        s = [msg.lower() for msg in s]
    if remove_accents:
        s = [deaccent(msg) for msg in s]
    if remove_urls:
        match_url_regexp = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        s = [re.sub(match_url_regexp, "", msg, flags=re.MULTILINE).strip() for msg in s]
    if remove_mentions:
        match_mentions_regexp = r"(@[a-zA-Z0-9_]+)"
        s = [
            re.sub(match_mentions_regexp, "", msg, flags=re.MULTILINE).strip()
            for msg in s
        ]
    if remove_twitter_cropend:
        match_croppedmsg_regexp = r"([^\s]+…)$"
        s = [
            re.sub(match_croppedmsg_regexp, "", msg, flags=re.MULTILINE).strip()
            for msg in s
        ]
    if remove_emojis:
        s = [demoji.replace(msg, "").strip() for msg in s]

    if remove_hashtags_frontend:
        if (not remove_urls) or (not remove_mentions):
            print(
                "Not all leading and ending hashtags might be removed because there might be mentions or urls"
            )
        match_hashtags_begin = r"(#\S+ ?)+"
        match_hashtags_end = r"(\S+# ?)+"
        match_hashtags_frontend = f"^({match_hashtags_begin})|^({match_hashtags_end})|({match_hashtags_begin})$|({match_hashtags_end})$"
        s = [re.sub(match_hashtags_frontend, "", msg).strip() for msg in s]
    if replace_newline_characters:
        match_escapes_regexp = r"(\n|\r)+"
        s = [
            re.sub(
                r"\s+", " ", re.sub(match_escapes_regexp, ". ", msg, flags=re.MULTILINE)
            ).strip()
            for msg in s
        ]
    if remove_punctuation:
        match_punctuations = r"[^\w\s]"
        s = [
            re.sub(r"\s+", " ", re.sub(match_punctuations, " ", msg)).strip()
            for msg in s
        ]
    if encapsulated:
        return s[0].strip()
    else:
        return s


@timeit
def prepare_dataset(dataset: Union[pd.Series, pd.DataFrame], min_size_txt: int = 30):
    """
    Create new columns of preprocessed texts from original text for distance comparison with 3 delta method
    Args:
        dataset (Union[pd.Series, pd.DataFrame]): dataframe or series containing a column "original" with the text. Optional: a column "language" can be given, otherwise language detection is implemented.
        min_size_txt (Optional[int], optional): size of text that should'nt taken into account for duplicate content because too small. If set to None, no text is removed. Defaults to {default_min_size}.
    Returns:
        dataset (pd.DataFrame): The same input dataset with new columns added (text_grapheme, text_to_embed, text_language_detect), containing the preprocessed texts for 3 delta method.
    """
    assert isinstance(
        dataset, (pd.Series, pd.DataFrame)
    ), "dataset must be a pd.Series or a pd.DataFrame"

    assert dataset.index.nunique() == len(
        dataset
    ), "dataset must be indexed with unique indices"

    assert all(
        [isinstance(i, str) for i in dataset.index]
    ), "dataset indices must be `str`"

    if isinstance(dataset, pd.DataFrame):
        assert (
            "original" in dataset.columns
        ), "when dataset is a pd.DataFrame, it must have a column named 'original'"

    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame("original")

    # text_grapheme is used for grapheme distance (Levenshtein)
    # this is the cleanest version with no spaces
    if "text_grapheme" not in dataset.columns:
        dataset["text_grapheme"] = [
            t.replace(" ", "")
            for t in preprocess_text(
                dataset["original"],
                lower=True,
                remove_accents=True,
                remove_urls=True,
                remove_mentions=True,
                remove_emojis=True,
                remove_hashtags_frontend=False,
                remove_twitter_cropend=False,
                replace_newline_characters=True,
                remove_punctuation=True,
            )
        ]

    # text_to_embed is used for semantic distance and embedded with USE
    # links are removed
    if "text_to_embed" not in dataset.columns:
        dataset["text_to_embed"] = preprocess_text(
            dataset["original"],
            lower=False,
            remove_accents=False,
            remove_urls=True,
            remove_mentions=True,
            remove_emojis=False,
            remove_hashtags_frontend=False,
            remove_twitter_cropend=False,
            replace_newline_characters=False,
            remove_punctuation=False,
        )
    # text_language_detect is used for fasttext
    # accents and emojis are kept as they provide interesting cues to language
    if ("language" not in dataset.columns) or (
        "text_language_detect" not in dataset.columns
    ):
        dataset["text_language_detect"] = preprocess_text(
            dataset["original"],
            lower=False,
            remove_accents=False,
            remove_urls=True,
            remove_mentions=True,
            remove_emojis=True,
            remove_hashtags_frontend=True,
            remove_twitter_cropend=False,
            replace_newline_characters=True,
            remove_punctuation=False,
        )
    print("Done.")
    print("")

    if min_size_txt is not None:
        print(
            f'Removing {(dataset["text_grapheme"].str.len() < min_size_txt).sum()} short texts over {len(dataset)} sentences...'
        )
        dataset = dataset.loc[dataset["text_grapheme"].str.len() >= min_size_txt]
        print("Done.")

    return dataset


@timeit
def compute_language(
    dataset: pd.DataFrame,
    fasttext_model=None,
    batch_size: int = 100,
    max_workers: int = 8,
):
    """
    Compute language detection in order to detect translation
    Args :
        dataset (pd.DataFrame): dataframe containing the column "text_language_detect" with the text to be analyzed
        fasttext_model (Optional[any], optional): optional, if another model than fasttext is to be used, otherwise, fasttext is uploaded. Defaults to None.
        batch_size (int, optional): batch size of text to be retrieved each step by parallelization. Defaults to 100.
        max_workers (int, optional): number of workers for parallelization. Defaults to 8.
    Returns:
        dataset (pd.DataFrame): The same input dataset with column 'language' added containing the results of language detection.
    """
    assert (
        "text_language_detect" in dataset.columns
    ), "you need to have a column text_language_detect to detect language"

    if fasttext_model is None:
        if os.path.exists("lid.176.ftz"):
            print("Loading fastext model from local file...")
            fasttext_model = fasttext.load_model("lid.176.ftz")
        else:
            print("Downloading fastext model from website and saving locally...")
            r = requests.get(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            )
            with open("lid.176.ftz", "wb") as f:
                f.write(r.content)
            fasttext_model = fasttext.load_model("lid.176.ftz")
        print("Done.\n")

    def process_chunk_fasttext(text_chunk, threshold=0.5):
        preds = fasttext_model.predict(text_chunk.tolist(), k=1)
        preds = [
            lng[0][-2:] if score[0] > threshold else ""
            for lng, score in zip(preds[0], preds[1])
        ]
        return preds

    batch_size = batch_size
    chunk_fasttext = thread_map(
        process_chunk_fasttext,
        grouper(dataset["text_language_detect"], batch_size),
        max_workers=max_workers,
        total=len(dataset) // batch_size,
    )

    dataset["language"] = np.concatenate(chunk_fasttext)
    return dataset


#############################
#### Compute Embeddings  ####
#############################


def download_USE(
    use_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
):
    use_model = hub.load(use_url)
    tf.saved_model.save(use_model, "use_model_kaggle")
    return use_model


@timeit
def compute_embeddings(df, batch_size: int = 100, max_workers: int = 8):
    """
    Compute embeddings for distance comparison
    Args:
        df (pd.DataFrame): dataframe containing the column "text_to_embed" with the text to be embedded
        batch_size (int, optional): batch size of text to be retrieved each step by parallelization. Defaults to 100.
        max_workers (int, optional): number of workers for parallelization. Defaults to 8.
    Returns:
        dataset (pd.DataFrame): A dataset with new columns added containing the results of embeddings computation.
    """
    assert "text_to_embed" in df.columns, print(
        "You need to compute text_to_embed columns"
    )
    use_model = download_USE()

    def process_chunk_use(text_chunk):
        return pd.DataFrame(
            use_model(text_chunk).numpy(),
            index=text_chunk.index,
            columns=[f"USE:{i}" for i in range(512)],
        )

    batch_size = batch_size
    chunk_use = thread_map(
        process_chunk_use,
        grouper(df["text_to_embed"], batch_size),
        max_workers=max_workers,
        total=len(df) // batch_size,
    )
    dataset = pd.concat([pd.concat(chunk_use, axis=0)], axis=1)
    dataset.index = df.index
    return dataset


@timeit
def create_index_cosine(df_embeddings: pd.DataFrame):
    """ "
    Create index with faiss for faster cosine distance computation
    Args:
        df_embeddings (pd.DataFrame): dataframe containing the embeddings
    Returns:
        index: A faiss index which can be used to compute cosine distances more efficiently
    """
    embeddings = df_embeddings.to_numpy()
    ids = list(df_embeddings.index)

    # cosine similarity index...
    vector_dimension = embeddings.shape[1]
    index_flat = faiss.IndexFlat(vector_dimension, faiss.METRIC_INNER_PRODUCT)
    # ...encapsulated in another index in order to have posts ids
    index = faiss.IndexIDMap(index_flat)

    # for cosine similarity, need of normalisation
    try:
        faiss.normalize_L2(embeddings)
    except:
        embeddings = embeddings.copy(order="C")
        faiss.normalize_L2(embeddings)
        print("C contiguous problem solved")

    # add embeddings & ids
    index.add_with_ids(embeddings, ids)
    return index


@timeit
def find_matches(
    df_embeddings_search: pd.DataFrame,
    index,
    threshold: float = 0.7,
    batch_size: int = 100,
    verbose=True,
):
    """
    Compute pairwise cosine similarity between all docs in index between a subset of docs and all docs in index
    Args :
        df_embeddings_search (pd.DataFrame): dataframe containing embeddings we want to find similarity with in the faiss index
        index: faiss index
        threshold (float, optional): threshold for similarity. Defaults to 0.7.
        batch_size (int, optional): number of vector per batch. Defaults to 100.
    Returns :
        matches (pd.DataFrame): A dataframe of pairs of duplicated texts with cosine score associated.
    """
    list_indices = []
    for i_batch in trange(
        0, len(df_embeddings_search), batch_size, disable=not verbose
    ):
        limits, distances, indices = index.range_search(
            df_embeddings_search.iloc[i_batch : i_batch + batch_size].to_numpy(),
            thresh=threshold,
        )
        for lim in range(len(limits) - 1):
            source = df_embeddings_search.index[i_batch + lim]
            for target, score in zip(
                indices[limits[lim] : limits[lim + 1]],
                distances[limits[lim] : limits[lim + 1]],
            ):
                if str(target) != str(source):  # doesn't match with its own embedding
                    list_indices.append([str(source), str(target), score])

    # create matches dataframe
    matches = pd.DataFrame(list_indices, columns=["source", "target", "score"])
    # drop duplicates because we have A-B and B-A
    matches["duplicates"] = matches.apply(
        lambda row: str(min(row["source"], row["target"]))
        + "-"
        + str(max(row["source"], row["target"])),
        axis=1,
    )
    matches = matches.drop_duplicates("duplicates")
    return matches


def find_matches_without_embeddings(
    df: pd.DataFrame,
    threshold_grapheme: float = 0.8,
    batch_size: int = 1000
):
    """
    Find potential matches using only grapheme-based similarity when embeddings are skipped.
    This creates pairwise comparisons using only text similarity for copy-pasta detection.
    
    Args:
        df (pd.DataFrame): dataframe containing text data with text_grapheme column
        threshold_grapheme (float): threshold for grapheme similarity to consider as potential match
        batch_size (int): batch size for processing to avoid memory issues
        
    Returns:
        matches (pd.DataFrame): A dataframe of pairs with dummy semantic scores
    """
    print("Finding matches using grapheme distance only (copy-pasta detection)...")
    
    indices = df.index.tolist()
    matches_list = []
    
    # Process in batches to avoid memory issues with large datasets
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        for idx1 in batch_indices:
            for idx2 in indices:
                if idx1 < idx2:  # Avoid duplicate pairs and self-comparison
                    text1 = df.loc[idx1, 'text_grapheme']
                    text2 = df.loc[idx2, 'text_grapheme']
                    
                    # Calculate grapheme similarity
                    lev_score = similarity_levenshtein((text1, text2))
                    
                    # Only keep pairs that have high grapheme similarity (potential copy-pasta)
                    # We use a stricter threshold since we're only looking for copy-pasta
                    if lev_score > threshold_grapheme:
                        # Use a dummy semantic score that will be ignored later
                        matches_list.append([str(idx1), str(idx2), 1.0])  # dummy score of 1.0
    
    # Create matches dataframe
    matches = pd.DataFrame(matches_list, columns=["source", "target", "score"])
    
    # Remove duplicates
    matches["duplicates"] = matches.apply(
        lambda row: str(min(row["source"], row["target"])) + "-" + str(max(row["source"], row["target"])),
        axis=1,
    )
    matches = matches.drop_duplicates("duplicates")
    
    print(f"Found {len(matches)} potential copy-pasta matches for further analysis")
    return matches


def similarity_levenshtein(pair):
    s1, s2 = pair
    return 1 - levenshtein(s1, s2) / max(len(s1), len(s2))


@timeit
def compute_duplicate_types(
    matches: pd.DataFrame,
    threshold_grapheme=0.693,
    threshold_language=0.715,
    threshold_semantic=0.85,
    skip_semantic=False,
):
    """
    Distinguish 3 different duplicate types: translation, rewording & copypasta
    Args :
        matches (pd.DataFrame): dataframe of pairs of texts containing text_grapheme_source and text_grapheme_target columns for detecting copypasta and language_source and language_target for detecting translation
        threshold_grapheme (float, optional): threshold to distinguish copypasta from rewording using levenshtein. Defaults to 0.693.
        threshold_language (float, optional): threshold to detect translation. Defaults to 0.715.
        threshold_semantic (float, optional): threshold to detect rewording. Defaults to 0.85.
        skip_semantic (bool, optional): if True, only copy-pasta detection is performed, ignoring semantic scores. Defaults to False.
    Returns :
        matches_strict (pd.DataFrame): dataframe containing 'copypasta', 'translation' and 'rewording' pairs of texts with score (cosine similarity from embeddings) and score_lev (score calculated from levenshtein) associated.
    """
    assert ("text_grapheme_source" in matches.columns) & (
        "text_grapheme_target" in matches.columns
    ), print(
        "You need text_grapheme_source and text_grapheme_target columns in dataframe for Levenstein"
    )

    if skip_semantic:
        # When skipping semantic analysis, only perform copy-pasta detection
        print("Semantic analysis skipped. Performing copy-pasta detection only.")
        
        # Initialize dup_type column
        matches["dup_type"] = "potential"
        
        # Handle case where there are no matches
        if len(matches) == 0:
            matches["score_lev"] = []
            matches_strict = matches[matches.dup_type == "copy-pasta"]
        else:
            # Compute Levenshtein scores for all matches
            score_lev_list = []
            for idx, row in matches.iterrows():
                score = similarity_levenshtein(
                    (row["text_grapheme_source"], row["text_grapheme_target"])
                )
                score_lev_list.append(score)
            
            matches["score_lev"] = score_lev_list
            
            # Only classify as copy-pasta and filter based on grapheme similarity
            matches.loc[matches.score_lev > threshold_grapheme, "dup_type"] = "copy-pasta"
            
            # Return only copy-pasta matches
            matches_strict = matches[matches.dup_type == "copy-pasta"]
        
    else:
        # Original full analysis with language detection
        assert ("language_source" in matches.columns) & (
            "language_target" in matches.columns
        ), print(
            "You need language_source and language_target columns in dataframe for language analysis"
        )

        matches["dup_type"] = "rewording"
        matches.loc[
            matches["language_source"] != matches["language_target"], "dup_type"
        ] = "translation"

        # Compute Levenshtein scores for all potential matches
        score_lev_list = []
        for idx, row in matches.iterrows():
            score = similarity_levenshtein(
                (row["text_grapheme_source"], row["text_grapheme_target"])
            )
            score_lev_list.append(score)
        
        matches["score_lev"] = score_lev_list
        
        # Classify as copy-pasta based on high grapheme similarity
        matches.loc[matches.score_lev > threshold_grapheme, "dup_type"] = "copy-pasta"

        # Original filtering logic using semantic scores
        matches_strict = matches[
            ((matches.score > threshold_semantic) & (matches.dup_type == "rewording"))
            | ((matches.score > threshold_language) & (matches.dup_type == "translation"))
            | (matches.dup_type == "copy-pasta")
        ]

    return matches_strict


def create_dataset_clusters(dataset: pd.DataFrame, edgelist: pd.DataFrame):
    """Give a cluster of duplicated content to all documents.

    None if no duplicated content was found for a document
    Args:
        dataset (pd.DataFrame): dataframe containing each document and same index used to create embeddings and faiss index.
        edgelist (pd.DataFrame): dataframe corresponding to pairs of texts and score associated
    Return:
        df_cluster (pd.DataFrame): dataframe with one row corresponding to one text and its cluster of duplicated content associated if it exists.
    """
    df_cluster = dataset.copy()
    consolidated_edgelist = edgelist.groupby(["source", "target"], as_index=False)[
        "score"
    ].max()
    clusters = list(
        nx.connected_components(nx.from_pandas_edgelist(consolidated_edgelist))
    )
    clusters.sort(key=len, reverse=True)
    for cluster_i, posts_indices in enumerate(clusters):
        df_cluster.loc[list(posts_indices), "cluster"] = cluster_i
    return df_cluster


def semantic_faiss(
    df: pd.DataFrame,
    min_size_txt: int = 0,
    df_embeddings_use: pd.DataFrame = None,
    embeddings_to_save: str = None,
    threshold_grapheme: float = 0.8,
    threshold_language: float = 0.715,
    threshold_semantic=0.9,
    remove_matches_same_user: str = None,
):
    """Apply end to end 3 delta methodology with faiss
    Args:
        df (pd.DataFrame): dataframe containing some columns :
            - original: text original
            - language (optional): language of each text. If not given, language detection is computed in order to detect translation
        min_size_txt (int): minimal size of text in order to apply 3 delta. if texts too short, removing document.
        df_embeddings_use (pd.DataFrame or str): embeddings dataframe already saved in order not to compute embeddings everytime. 
                                                 If "skip", semantic distance computation is skipped entirely.
        embeddings_to_save (str): name of pickle to save the embeddings if the user wants to save the embeddings.
        threshold_grapheme (float): threshold to detect copypasta with levenshtein on matches found with faiss. Defaults to 0.693.
        threshold_language (float): threshold to find matches between 2 documents for translation. Defaults to 0.715.
        threshold_semantic (float): threshold to find matches between 2 documents for rewording. Defaults to 0.85.
    Return:
        matches (pd.DataFrame): dataframe containing pairs of text detected as duplicate contents from 3 delta
        df_cluster (pd.DataFrame): initial dataframe 'df' with its cluster of duplicated content associated if it exists.
    """

    df = prepare_dataset(df, min_size_txt=min_size_txt)

    if "language" not in df.columns:
        print("language detection")
        df = compute_language(df)

    # Check if semantic distance should be skipped
    skip_semantic = (type(df_embeddings_use) == str and df_embeddings_use == "skip")
    
    if skip_semantic:
        print("Skipping semantic distance computation as requested...")
        res = find_matches_without_embeddings(df, threshold_grapheme)
    else:
        if df_embeddings_use is None:
            df_embeddings_use = compute_embeddings(df)
            if embeddings_to_save is not None:
                df_embeddings_use.to_pickle(f"{embeddings_to_save}.pkl")

        index_faiss = create_index_cosine(df_embeddings_use)

        threshold_faiss = min(threshold_language, threshold_semantic)
        res = find_matches(df_embeddings_use, index_faiss, threshold=threshold_faiss)

    if remove_matches_same_user is not None:
        columns_join = [
            remove_matches_same_user,
            "language",
            "text_to_embed",
            "text_grapheme",
        ]
    else:
        columns_join = ["language", "text_to_embed", "text_grapheme"]
    matches = res.merge(
        df[columns_join].add_suffix("_source"),
        left_on="source",
        right_index=True,
        how="left",
    ).merge(
        df[columns_join].add_suffix("_target"),
        left_on="target",
        right_index=True,
        how="left",
    )
    matches = compute_duplicate_types(
        matches,
        threshold_grapheme=threshold_grapheme,
        threshold_language=threshold_language,
        threshold_semantic=threshold_semantic,
        skip_semantic=skip_semantic,
    )

    if remove_matches_same_user is not None:
        matches = matches[
            matches[remove_matches_same_user + "_source"]
            != matches[remove_matches_same_user + "_target"]
        ]

    df_clusters = create_dataset_clusters(df, matches)

    return matches, df_clusters

### STUFF ADDED BY SAVA

def prepare_input_data(dataframe, embeddings=None, model=None):
    """
    Prepares a filtered DataFrame and its corresponding embedding DataFrame.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        Input DataFrame expected to contain a 'tweet' column.
    embeddings : np.ndarray, pd.Series, or None, optional
        Precomputed embeddings for the 'tweet' column. If None, embeddings
        will be computed using the provided `model`.
    model : object, optional
        A model with an `.encode()` method that converts a list of texts
        into their embedding vectors. Required if `embeddings` is None.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the 'tweet' column and a default 'language' column.
    pd.DataFrame
        A DataFrame of corresponding embeddings with aligned index.
    """
    tweets_df = dataframe[['tweet']].copy()

    if embeddings is None:
        if model is None:
            raise ValueError("Either `embeddings` must be provided or `model` must be specified.")
        embeddings = model.encode(tweets_df['tweet'].tolist())

    if isinstance(embeddings, pd.Series):
        embeddings.index = embeddings.index.astype(str)
        embeddings = embeddings.loc[tweets_df.index].values
        embeddings_df = pd.DataFrame(np.stack(embeddings, dtype=np.float32), index=tweets_df.index)
    else:
        embeddings_df = pd.DataFrame(embeddings, index=tweets_df.index)

    tweets_df['language'] = 'fr'

    return tweets_df, embeddings_df
