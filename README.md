# Astrodetection

Astrodetection is a Python library designed for detecting astroturfing clues from lists of posts (mainly on X up to now, but not exclusively)


## Installation

### Pip

```bash
pip install "astrodetection[standard]"
```

or

```bash
pip install "astrodetection[light]"
```

### Conda

1. Use the YAML file to configure the environment with conda:

   ```bash
   conda create -n astrodetection_env
   conda activate astrodetection_env
   conda env update -f environment_standard.yml
   ```

**Note:** the ```environment_standard.yml``` configuration file uses FAISS and Fasttext libraries for [VIGINUM D3LTA implementation](https://github.com/VIGINUM-FR/D3lta)

**If you have compatibility issues, prefer ```environment_light.yml``` and use ```astrodetection_light``` module

## Usage

You can import directly the main functions:

```python
from astrodetection import semantic_faiss, prepare_input_data, compute_bot_likelihood_metrics, create_network, copypasta_score_hub
```

Or use them directly:

```python
import glob
import pandas as pd
import os
import numpy as np
import astrodetection

# Load a single JSON file into a DataFrame
file = "file_path"  # Select the first file
df = pd.read_json(file)
df.index = df.index.astype(str)  # Compatibility with d3lta

# Preprocess the DataFrame
df = df[df['tweet'].str.len() > 100]
df = df[df['username'] != 'grok']
df.index = df.index.astype(str)

# Compute matches and scores
df_filtered, df_emb = astrodetection.prepare_input_data(df, embeddings=df['emb'])

matches, df_cluster = astrodetection.semantic_faiss(
    df_filtered.rename(columns={'tweet': 'original'}),
    min_size_txt=0,
    df_embeddings_use=df_emb,
    threshold_grapheme=0.8,
    threshold_language=0.715,
    threshold_semantic=0.9
) #function taken from D3LTA 

# Create the raw post-similarity network and measure its largest component
network = astrodetection.create_network(matches, df, return_sigma=False)
copypasta_hub = astrodetection.copypasta_score_hub(network, df)

# The same indicator is also available in the combined metrics
scores = astrodetection.compute_bot_likelihood_metrics(
    df,
    matches=matches,
    G_copypasta=network,
)

# Create the interactive visualization when needed
network_viz = astrodetection.create_network(matches, df)
```

# New changes

1. _`semantic_faiss`_ function can now take detect only copypastas based on levenshtein distance, ignoring embeddings, if "skip" is passed as argument in _df_embeddings_use_ field.

2. _`compute_bot_likelihood_metrics`_ function can now take columns' names as arguments for more customization

3. _`compute_bot_likelihood_metrics`_ now returns _`high_following_followers_ratio (%)`_, the share of rows whose following/followers ratio exceeds _fw_ratio_threshold_ (default 10).

4. _`copypasta_score_hub`_ returns the percentage of all original posts belonging to the largest connected component of a post-similarity network. By default it retains every duplicate type and every edge already present in the network.
