# Astrodetection

Astrodetection is a Python library designed for detecting astroturfing clues from lists of posts (mainly on X up to now, but not exclusively)


## Installation

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
from astrodetection import semantic_faiss, prepare_input_data, compute_bot_likelihood_metrics, create_network
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

scores = astrodetection.compute_bot_likelihood_metrics(df, matches=matches)

# Create a network
network = astrodetection.create_network(matches, df)
```

