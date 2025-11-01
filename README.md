# CLIR-News

This is a Chinese-English cross-language news retrieval system. We use vec-mapping techniques to align word2vec of different languages, and integrate them with information retrieval pipelines.

## Project Proposal

[project proposal](./SI650_Project_Proposal_G9.pdf)

## Setup

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Download Pre-trained Word2Vec Embeddings

run:

```bash
chmod +x ./get_resources.sh

./get_resources.sh
```

and wait for the script to finish.

## Using the Aligned Word2Vec Embeddings

The `pretrained_word2vec/zh-aligned/` folder contains Chinese word embeddings that have been aligned to the English GoogleNews vector space. 

**Important**: None of these aligned models contain English embeddings themselves, so you'll need to load both the Chinese (aligned) and English embeddings separately and implement logic to determine which embedding to use based on the input language.

The aligned embeddings are stored in timestamped directories (e.g., `renmin-google/20251023-021029/`) and are in compressed Word2Vec binary format (`.bin.gz`).

### Chinese Embeddings Details

The Chinese embeddings are sourced from the [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) repository with the following preprocessing:

- **Word segmentation**: [HanLP (v1.5.3)](https://github.com/hankcs/HanLP) is used for tokenization
- **Character conversion**: Traditional Chinese characters are converted to Simplified Chinese using [Open Chinese Convert (OpenCC)](https://github.com/BYVoid/OpenCC)

**Recommendation**: For best performance, use the same toolkit (HanLP v1.5.3 + OpenCC) when preprocessing your Chinese text queries.

### English Embeddings Details

The English embeddings are from [Google's pre-trained word2vec vectors](https://code.google.com/archive/p/word2vec/):

- **Multi-word phrases**: The model contains multi-word phrases connected with underscores (e.g., `New_York_Times`, `machine_learning`)

**Recommendation**: Use gensim's built-in phrase detector (`gensim.models.phrases.Phrases`) to preprocess your English text, as it's implemented in the same way as the original word2vec phrase detection

### Usage Example

Here's a basic example of how to load and use both embeddings:

```python
from gensim.models import KeyedVectors

# Load aligned Chinese embeddings
zh_model = KeyedVectors.load_word2vec_format(
    'pretrained_word2vec/zh-aligned/renmin-google/20251023-021029/sgns.renmin.bin.gz',
    binary=True
)

# Load English embeddings
en_model = KeyedVectors.load_word2vec_format(
    'pretrained_word2vec/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

def get_embedding(word, language='auto'):
    """
    Get embedding for a word, automatically detecting language if needed.

    Args:
        word: The word to get embedding for
        language: 'zh', 'en', or 'auto' for automatic detection

    Returns:
        numpy array of the word embedding
    """
    if language == 'auto':
        # Simple heuristic: check if word contains Chinese characters
        language = 'zh' if any('\u4e00' <= char <= '\u9fff' for char in word) else 'en'

    if language == 'zh':
        return zh_model[word] if word in zh_model else None
    else:
        return en_model[word] if word in en_model else None

# Example usage
zh_vec = get_embedding('中国', 'zh')  # Chinese word
en_vec = get_embedding('machine_learning', 'en')  # English phrase
auto_vec = get_embedding('北京')  # Auto-detect (Chinese)
```

### Files Structure

```
pretrained_word2vec/zh-aligned/
├── renmin-google/
│   └── 20251023-021029/
│       └── sgns.renmin.bin.gz
└── merge-google/
    └── 20251023-023520/
        └── sgns.merge.bin.gz
```