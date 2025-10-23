"""
Vector Mapping for Cross-Lingual Word Embedding Alignment
Architecture for aligning Chinese and English word embeddings using Procrustes analysis.
"""

import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Set, List, Optional, Tuple, Union
import re
from opencc import OpenCC
import warnings
import torch
import os

class EmbeddingLoader:
    """
    Loads and processes word embeddings from various formats.

    This class maintains state and ensures only one embedding file is loaded per instance.
    Supports Word2Vec binary and text formats with automatic compression handling.

    Examples:
        >>> # Load embeddings
        >>> loader = EmbeddingLoader()
        >>> loader.load_word2vec('GoogleNews.bin.gz')
        >>> print(len(loader))  # Vocabulary size

        >>> # Filter and process
        >>> loader.filter_vocabulary(common_words)
        >>> matrix = loader.get_embedding_matrix(['cat', 'dog'])

        >>> # Check word existence
        >>> if 'cat' in loader:
        >>>     vec = loader.get_embedding('cat')
    """

    def __init__(self):
        """Initialize the loader with empty state."""
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.source_file: Optional[str] = None
        self.embedding_dim: Optional[int] = None

    def _ensure_not_loaded(self):
        """
        Check that no embeddings have been loaded yet.

        Raises:
            RuntimeError: If embeddings have already been loaded
        """
        if self.embeddings is not None:
            raise RuntimeError(
                f"Embeddings already loaded from '{self.source_file}'. "
                f"Use reload() to clear current embeddings, or create a new "
                f"EmbeddingLoader instance."
            )

    def reload(self) -> 'EmbeddingLoader':
        """
        Clear current embeddings and reset loader to initial state.

        This allows reusing the same EmbeddingLoader instance to load
        different embedding files without creating a new instance.

        Returns:
            self (for method chaining)

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('file1.bin')
            >>> print(len(loader))  # e.g., 100000

            >>> # Reload with different file
            >>> loader.reload()
            >>> loader.load_word2vec('file2.bin')
            >>> print(len(loader))  # e.g., 50000

            >>> # Method chaining works too
            >>> loader.reload().load_word2vec('file3.bin')
        """
        self.embeddings = None
        self.source_file = None
        self.embedding_dim = None
        return self

    def _ensure_loaded(self):
        """
        Check that embeddings have been loaded.

        Raises:
            RuntimeError: If no embeddings have been loaded yet
        """
        if self.embeddings is None:
            raise RuntimeError(
                "No embeddings loaded. Call load_word2vec() first."
            )

    def load_word2vec(
        self,
        filepath: str,
        format: str = 'auto',
        max_vocab: Optional[int] = None
    ) -> 'EmbeddingLoader':
        """
        Load pre-trained word embeddings from a Word2Vec format file.

        Args:
            filepath: Path to the embedding file (supports .bin, .bin.gz, .vec, .txt, .bz2)
            format: Loading format - 'auto' (default), 'binary', or 'text'
                   'auto': Auto-detect from file extension
                   'binary': Force Word2Vec binary format
                   'text': Force text format
            max_vocab: Maximum number of words to load (useful for testing). None = load all

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If embeddings already loaded
            FileNotFoundError: If file doesn't exist

        Note:
            Vectors are stored in their original form without preprocessing.
            Use VectorPreprocessor for normalization, centering, or other transformations.

        Examples:
            >>> # Auto-detect format (recommended)
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('GoogleNews.bin.gz')
            >>>
            >>> # Load subset for testing
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('big.bin', max_vocab=10000)
            >>>
            >>> # Explicit format
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('custom.vec', format='text')
        """
        self._ensure_not_loaded()

        if format == 'auto':
            lower_path = filepath.lower()
            if any(ext in lower_path for ext in ['.bin.gz', '.bin']):
                binary = True
            else:
                binary = False
        elif format == 'binary':
            binary = True
        elif format == 'text':
            binary = False
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'auto', 'binary', or 'text'")

        try:
            kv = KeyedVectors.load_word2vec_format(
                filepath,
                binary=binary,
                limit=max_vocab,
                no_header=False if binary else None
            )

            embeddings = {word: kv[word].copy() for word in kv.key_to_index}

            self.embeddings = embeddings
            self.source_file = filepath
            self.embedding_dim = kv.vector_size if embeddings else None

            return self

        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading embeddings from {filepath}: {str(e)}")

    def filter_vocabulary(
        self,
        words: Set[str]
    ) -> 'EmbeddingLoader':
        """
        Filter stored embeddings to keep only words in the specified vocabulary.

        Args:
            words: Set of words to keep

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('large.bin')
            >>> # Keep only words that have dictionary translations
            >>> dict_words = {'cat', 'dog', 'house'}
            >>> loader.filter_vocabulary(dict_words)
            >>> print(len(loader))  # 3 (or fewer if some words not in embeddings)
        """
        self._ensure_loaded()

        self.embeddings = {word: vec for word, vec in self.embeddings.items() if word in words}

        return self

    def get_embedding_matrix(
        self,
        words: List[str],
        return_words: bool = True,
        strict: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert stored embeddings to aligned 2D matrix for specified words.

        Creates a matrix where row i corresponds to words[i], ensuring
        alignment for parallel training data (e.g., source-target pairs).

        ⚠️ IMPORTANT: Words without embeddings are silently skipped unless strict=True.
        Always use return_words=True to get the list of actually included words.

        Args:
            words: Ordered list of words (defines row order in output matrix)
            return_words: If True (default), return (matrix, valid_words) tuple.
            strict: If True, raise ValueError if any words are missing embeddings.
                   If False (default), skip missing words silently.

        Returns:
            If return_words=True: (matrix, valid_words) tuple where:
                - matrix: shape (N, embedding_dim) - only valid words included
                - valid_words: list of N words actually included in matrix
            If return_words=False: Only the matrix (DEPRECATED)

        Raises:
            RuntimeError: If no embeddings have been loaded
            ValueError: If none of the words are found in embeddings
            ValueError: If strict=True and any words are missing

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')

            >>> # Recommended usage - get both matrix and valid words
            >>> words = ['cat', 'dog', 'nonexistent']
            >>> matrix, valid = loader.get_embedding_matrix(words)
            >>> matrix.shape  # (2, 300) - only 2 words found
            >>> valid  # ['cat', 'dog'] - shows which words were included

            >>> # Strict mode - error if any words missing
            >>> matrix, valid = loader.get_embedding_matrix(words, strict=True)
            ValueError: 1 words not found in embeddings: nonexistent

            >>> # Backward compatible (DEPRECATED - avoid this!)
            >>> matrix = loader.get_embedding_matrix(words, return_words=False)
            >>> matrix.shape  # (2, 300) - but you don't know which words!
        """
        self._ensure_loaded()

        valid_words = [w for w in words if w in self.embeddings]
        missing_words = [w for w in words if w not in self.embeddings]

        if strict and missing_words:
            missing_str = ', '.join(missing_words[:5])  # Show first 5
            if len(missing_words) > 5:
                missing_str += f', ... ({len(missing_words) - 5} more)'
            raise ValueError(
                f"{len(missing_words)} word(s) not found in embeddings: {missing_str}"
            )

        if len(valid_words) == 0:
            raise ValueError("None of the provided words found in embeddings")

        vectors = np.vstack([self.embeddings[word] for word in valid_words])

        if return_words:
            return vectors, valid_words
        else:
            # DEPRECATED: Returning only matrix can cause alignment bugs
            warnings.warn(
                "Returning only the embedding matrix (return_words=False) is deprecated "
                "and can cause alignment bugs. Please use return_words=True to get both "
                "the matrix and the list of valid words.",
                DeprecationWarning,
                stacklevel=2
            )
            return vectors

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get the loaded embeddings dictionary.

        Returns:
            Dictionary mapping words to embedding vectors

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> emb_dict = loader.get_embeddings()
            >>> print(emb_dict['cat'])  # numpy array
        """
        self._ensure_loaded()
        return self.embeddings

    def get_vocabulary(self) -> Set[str]:
        """
        Get the set of words in the loaded embeddings.

        Returns:
            Set of words (vocabulary)

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> vocab = loader.get_vocabulary()
            >>> 'cat' in vocab  # True
        """
        self._ensure_loaded()
        return set(self.embeddings.keys())

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a specific word.

        Args:
            word: The word to look up

        Returns:
            Embedding vector for the word

        Raises:
            RuntimeError: If no embeddings have been loaded
            KeyError: If word not found in embeddings

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> vec = loader.get_embedding('cat')
            >>> vec.shape  # (300,)
        """
        self._ensure_loaded()
        return self.embeddings[word]

    def __len__(self) -> int:
        """
        Return the vocabulary size.

        Returns:
            Number of words in the loaded embeddings

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> print(len(loader))  # 100000
        """
        self._ensure_loaded()
        return len(self.embeddings)

    def __contains__(self, word: str) -> bool:
        """
        Check if a word exists in the loaded embeddings.

        Args:
            word: The word to check

        Returns:
            True if word exists, False otherwise

        Raises:
            RuntimeError: If no embeddings have been loaded

        Examples:
            >>> loader = EmbeddingLoader()
            >>> loader.load_word2vec('embeddings.bin')
            >>> 'cat' in loader  # True
            >>> 'asdfghjkl' in loader  # False
        """
        self._ensure_loaded()
        return word in self.embeddings


class DictionaryParser:
    """
    Parses bilingual dictionaries from various formats (CC-CEDICT, MUSE)
    with support for traditional/simplified Chinese conversion.

    This class maintains state and ensures only one dictionary is loaded per instance.
    Use OpenCC for automatic traditional/simplified Chinese conversion.

    Examples:
        >>> # Parse CEDICT with only simplified Chinese
        >>> parser = DictionaryParser()
        >>> parser.parse_cedict('cedict_ts.u8', include_traditional=False)
        >>> print(len(parser.get_pairs()))  # Number of translation pairs

        >>> # Parse with both traditional and simplified
        >>> parser2 = DictionaryParser()
        >>> parser2.parse_cedict('cedict_ts.u8', include_traditional=True)
        >>> parser2.filter_by_vocabulary(zh_vocab, en_vocab)
        >>> parser2.save('filtered_dict.txt')

        >>> # Parse MUSE format
        >>> parser3 = DictionaryParser()
        >>> parser3.parse_muse_format('zh-en.txt', include_traditional=False)
    """

    def __init__(self):
        """Initialize the parser with OpenCC converters and empty state."""
        self.pairs: Optional[List[Tuple[str, str]]] = None
        self.source_file: Optional[str] = None
        self.format_type: Optional[str] = None

        try:
            self._t2s_converter = OpenCC('t2s')  # Traditional to Simplified
            self._s2t_converter = OpenCC('s2t')  # Simplified to Traditional
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCC converters: {str(e)}")

    def _ensure_not_loaded(self):
        """
        Check that no dictionary has been loaded yet.

        Raises:
            RuntimeError: If a dictionary has already been loaded
        """
        if self.pairs is not None:
            raise RuntimeError(
                f"Dictionary already loaded from '{self.source_file}' "
                f"(format: {self.format_type}). Create a new DictionaryParser instance "
                f"to load a different dictionary."
            )

    def _contains_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.

        Args:
            text: Text to check for Chinese characters

        Returns:
            True if text contains Chinese characters, False otherwise

        Examples:
            >>> parser = DictionaryParser()
            >>> parser._contains_chinese('hello')
            False
            >>> parser._contains_chinese('hello中国')
            True
        """
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _is_single_word(self, text: str) -> bool:
        """
        Check if text represents a single word (no whitespace).

        Args:
            text: Text to check

        Returns:
            True if text is non-empty and contains no whitespace
        """
        text = text.strip()
        return bool(text) and ' ' not in text

    def _extract_single_word(self, text: str) -> Optional[str]:
        """
        Extract a single English word from definition text.

        Processing strategy:
        1. Clean: Remove parentheses (often contain annotations)
        2. Split: Try semicolons first (major separators), then commas
        3. Find: Look for the first part that is a single word
        4. Fallback: If no separators, check if entire text is single word

        Args:
            text: English definition text to process

        Returns:
            Single English word if extraction successful, None otherwise

        Examples:
            >>> parser = DictionaryParser()
            >>> parser._extract_single_word('cat (animal); feline')
            'feline'
            >>> parser._extract_single_word('run, move quickly')
            'run'
            >>> parser._extract_single_word('very good; extremely nice')
            None
            >>> parser._extract_single_word('cat')
            'cat'
        """
        # Clean: Remove parenthetical annotations
        text = text.replace('(', '').replace(')', '')

        # Split: Try separators in order of priority (semicolon > comma)
        for separator in [';', ',']:
            if separator in text:
                parts = [part.strip() for part in text.split(separator)]
                # Find: Return first single-word part
                for part in parts:
                    if self._is_single_word(part):
                        return part
                return None

        # Fallback: Check if entire text (no separators) is a single word
        return text.strip() if self._is_single_word(text) else None

    def _convert_with_variants(
        self,
        chinese_text: str,
        include_traditional: bool
    ) -> List[str]:
        """
        Convert Chinese text to simplified (and optionally traditional) forms.

        Args:
            chinese_text: Original Chinese text
            include_traditional: If True, return both forms; if False, only simplified

        Returns:
            List of Chinese variants (1 or 2 elements). Duplicates are removed.

        Examples:
            >>> parser = DictionaryParser()
            >>> parser._convert_with_variants('中国', False)
            ['中国']
            >>> parser._convert_with_variants('中國', True)
            ['中国', '中國']  # Both forms
            >>> parser._convert_with_variants('中国', True)
            ['中国']  # Only one form if already simplified
        """
        simplified = self._t2s_converter.convert(chinese_text)

        if not include_traditional:
            return [simplified]

        traditional = self._s2t_converter.convert(chinese_text)

        if simplified != traditional:
            return [simplified, traditional]
        else:
            return [simplified]

    def parse_cedict(
        self,
        filepath: str,
        include_traditional: bool = False
    ) -> 'DictionaryParser':
        """
        Parse CC-CEDICT format dictionary file.

        CC-CEDICT format:
            Traditional Simplified [pin1 yin1] /English 1/English 2/...
            Example: 中國 中国 [Zhong1 guo2] /China/Middle Kingdom/

        Args:
            filepath: Path to CC-CEDICT file (usually .u8 extension)
            include_traditional: If True, generate pairs for both simplified and
                               traditional forms (when different)

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If dictionary already loaded
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict_ts.u8')
            >>> pairs = parser.get_pairs()
            >>> pairs[0]  # ('中国', 'China')
        """
        self._ensure_not_loaded()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"CEDICT file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading CEDICT file {filepath}: {str(e)}")

        pairs = []

        # CEDICT format: Traditional Simplified [pinyin] /def1/def2/.../
        # Example: 中國 中国 [Zhong1 guo2] /China/Middle Kingdom/
        cedict_pattern = re.compile(
            r'^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/$'
        )

        for line in lines:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            match = cedict_pattern.match(line)
            if not match:
                continue

            traditional, simplified, pinyin, definitions = match.groups()

            def_list = [d.strip() for d in definitions.split('/') if d.strip()]

            if len(def_list) != 1:
                continue

            english = def_list[0]

            if self._contains_chinese(english):
                continue

            english_word = self._extract_single_word(english)
            if english_word is None:
                continue

            chinese_variants = self._convert_with_variants(simplified, include_traditional)

            for chinese in chinese_variants:
                if ' ' not in chinese.strip():
                    pairs.append((chinese, english_word))

        self.pairs = pairs
        self.source_file = filepath
        self.format_type = 'cedict'

        return self

    def parse_muse_format(
        self,
        filepath: str,
        include_traditional: bool = False
    ) -> 'DictionaryParser':
        """
        Parse MUSE (Facebook) format dictionary file.

        MUSE format (tab or space separated):
            source_word target_word
            Example: 中国 China

        Args:
            filepath: Path to MUSE format dictionary file
            include_traditional: If True, generate pairs for both simplified and
                               traditional forms (when different)

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If dictionary already loaded
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_muse_format('zh-en.txt')
            >>> pairs = parser.get_pairs()
        """
        self._ensure_not_loaded()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"MUSE dictionary file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading MUSE file {filepath}: {str(e)}")

        pairs = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            parts = re.split(r'\s+', line)

            if len(parts) < 2:
                continue

            chinese = parts[0]
            english = parts[1]

            chinese_variants = self._convert_with_variants(chinese, include_traditional)

            for chinese_variant in chinese_variants:
                pairs.append((chinese_variant, english))

        self.pairs = pairs
        self.source_file = filepath
        self.format_type = 'muse'

        return self

    def filter_by_vocabulary(
        self,
        zh_vocab: Set[str],
        en_vocab: Set[str],
        warn_threshold: float = 0.5
    ) -> 'DictionaryParser':
        """
        Filter stored translation pairs to only include words present in both vocabularies.

        This is essential for training: ensures every word pair has embeddings available.

        ⚠️ If more than warn_threshold (default 50%) of pairs are dropped, a warning is issued.

        Args:
            zh_vocab: Set of Chinese words that have embeddings
            en_vocab: Set of English words that have embeddings
            warn_threshold: Fraction of pairs (0.0-1.0) that when exceeded triggers warning.
                          Set to 1.0 to disable warnings. Default: 0.5 (50%)

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If no dictionary has been loaded yet

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> print(len(parser.get_pairs()))  # 100000

            >>> parser.filter_by_vocabulary(zh_vocab, en_vocab)
            >>> # Warning: Filtered out 95.2% of pairs (95234/100000).
            >>> #          Only 4766 pairs remain.
            >>> print(len(parser.get_pairs()))  # 4766

            >>> # Suppress warnings
            >>> parser.filter_by_vocabulary(zh_vocab, en_vocab, warn_threshold=1.0)
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        original_count = len(self.pairs)

        filtered_pairs = [
            (zh, en) for zh, en in self.pairs
            if zh in zh_vocab and en in en_vocab
        ]

        new_count = len(filtered_pairs)
        dropped_count = original_count - new_count
        drop_rate = dropped_count / original_count if original_count > 0 else 0

        if drop_rate > warn_threshold:
            warnings.warn(
                f"Filtered out {drop_rate*100:.1f}% of pairs ({dropped_count}/{original_count}). "
                f"Only {new_count} pairs remain. "
                f"Consider using embeddings with larger vocabulary or a different dictionary.",
                UserWarning,
                stacklevel=2
            )

        self.pairs = filtered_pairs

        return self

    def save(self, filepath: str) -> None:
        """
        Save parsed dictionary pairs to a tab-separated file.

        Output format:
            chinese<tab>english<newline>

        Args:
            filepath: Path where to save the dictionary

        Raises:
            RuntimeError: If no dictionary has been loaded

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> parser.filter_by_vocabulary(zh_vocab, en_vocab)
            >>> parser.save('filtered_dict.txt')
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for chinese, english in self.pairs:
                    f.write(f"{chinese}\t{english}\n")
        except Exception as e:
            raise RuntimeError(f"Error saving dictionary to {filepath}: {str(e)}")

    def get_pairs(self) -> List[Tuple[str, str]]:
        """
        Get the loaded translation pairs.

        Returns:
            List of (chinese, english) tuples

        Raises:
            RuntimeError: If no dictionary has been loaded

        Examples:
            >>> parser = DictionaryParser()
            >>> parser.parse_cedict('cedict.u8')
            >>> pairs = parser.get_pairs()
            >>> print(pairs[0])  # ('中国', 'China')
        """
        if self.pairs is None:
            raise RuntimeError(
                "No dictionary loaded. Call parse_cedict() or parse_muse_format() first."
            )

        return self.pairs


class VectorPreprocessor:
    """
    Preprocessor for embedding vectors using MUSE normalization utilities.

    This class wraps MUSE's normalize_embeddings() function to provide
    L2 normalization and centering operations on numpy arrays.

    Examples:
        >>> import numpy as np
        >>> X = np.random.randn(100, 300)
        >>>
        >>> # Normalize to unit length
        >>> X_norm = VectorPreprocessor.normalize(X)
        >>>
        >>> # Center at origin
        >>> X_centered = VectorPreprocessor.center(X)
        >>>
        >>> # Both (recommended for alignment)
        >>> X_processed = VectorPreprocessor.center_and_normalize(X)
    """

    @staticmethod
    def _to_torch(X: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.

        Args:
            X: Numpy array of shape (n_samples, n_features)

        Returns:
            Float tensor on CPU
        """
        import torch
        return torch.from_numpy(X.copy()).float()

    @staticmethod
    def _to_numpy(X_torch: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.

        Args:
            X_torch: Torch tensor

        Returns:
            Numpy array
        """
        return X_torch.cpu().numpy()

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """
        L2 normalize each vector to unit length.

        Wraps MUSE normalize_embeddings() with types='renorm'.
        Each row is divided by its L2 norm.

        Args:
            X: Numpy array of shape (n_samples, n_features)

        Returns:
            Normalized array of same shape

        Examples:
            >>> X = np.array([[3, 4], [5, 12]])
            >>> X_norm = VectorPreprocessor.normalize(X)
            >>> np.allclose(np.linalg.norm(X_norm, axis=1), 1.0)
            True
        """
        from MUSE.src.utils import normalize_embeddings
        import torch

        X_torch = VectorPreprocessor._to_torch(X)
        normalize_embeddings(X_torch, types='renorm')
        return VectorPreprocessor._to_numpy(X_torch)

    @staticmethod
    def center(X: np.ndarray) -> np.ndarray:
        """
        Center vectors by subtracting the mean vector.

        Wraps MUSE normalize_embeddings() with types='center'.
        The mean vector is subtracted from all rows.

        Args:
            X: Numpy array of shape (n_samples, n_features)

        Returns:
            Centered array of same shape

        Examples:
            >>> X = np.random.randn(100, 300)
            >>> X_centered = VectorPreprocessor.center(X)
            >>> np.allclose(X_centered.mean(axis=0), 0, atol=1e-6)
            True
        """
        from MUSE.src.utils import normalize_embeddings
        import torch

        X_torch = VectorPreprocessor._to_torch(X)
        normalize_embeddings(X_torch, types='center')
        return VectorPreprocessor._to_numpy(X_torch)

    @staticmethod
    def center_and_normalize(X: np.ndarray) -> np.ndarray:
        """
        Center then normalize vectors (recommended preprocessing).

        Wraps MUSE normalize_embeddings() with types='center,renorm'.
        First centers by subtracting mean, then L2 normalizes.

        Args:
            X: Numpy array of shape (n_samples, n_features)

        Returns:
            Processed array of same shape

        Examples:
            >>> X = np.random.randn(100, 300)
            >>> X_processed = VectorPreprocessor.center_and_normalize(X)
            >>> # Check normalized
            >>> np.allclose(np.linalg.norm(X_processed, axis=1), 1.0)
            True
        """
        from MUSE.src.utils import normalize_embeddings
        import torch

        X_torch = VectorPreprocessor._to_torch(X)
        normalize_embeddings(X_torch, types='center,renorm')
        return VectorPreprocessor._to_numpy(X_torch)


class SimpleParams:
    """
    Lightweight parameter object for MUSE compatibility.

    MUSE functions expect a params object with specific attributes.
    This class provides a simple way to create such objects without
    importing full MUSE argument parsers.

    Examples:
        >>> params = SimpleParams(
        ...     cuda=False,
        ...     dico_method='csls_knn_10',
        ...     dico_build='S2T',
        ...     dico_max_rank=10000
        ... )
        >>> params.cuda
        False
        >>> params.dico_method
        'csls_knn_10'
    """

    def __init__(self, **kwargs):
        """
        Initialize params with keyword arguments.

        Common parameters for MUSE:
            cuda (bool): Use GPU if available (default: auto-detect)
            emb_dim (int): Embedding dimension (default: 300)
            src_lang (str): Source language code (default: 'zh')
            tgt_lang (str): Target language code (default: 'en')
            normalize_embeddings (str): Normalization type (default: 'center,renorm')
            dico_method (str): Dictionary building method (default: 'csls_knn_10')
            dico_build (str): Dictionary direction (default: 'S2T')
            dico_max_rank (int): Max rank for dictionary (default: 10000)
            dico_max_size (int): Max dictionary size (default: 0 = unlimited)
            dico_min_size (int): Min dictionary size (default: 0)
            dico_threshold (float): Confidence threshold (default: 0)
            map_id_init (bool): Initialize mapping to identity (default: True)
            exp_path (str): Experiment path for saving (default: './dumped')
            exp_name (str): Experiment name (default: 'debug')
            exp_id (str): Experiment ID (default: '')

        Args:
            **kwargs: Arbitrary keyword arguments to set as attributes

        Examples:
            >>> # Create params with defaults
            >>> params = SimpleParams()
            >>> params.cuda = torch.cuda.is_available()
            >>>
            >>> # Create with custom values
            >>> params = SimpleParams(
            ...     emb_dim=300,
            ...     dico_method='csls_knn_10',
            ...     cuda=False
            ... )
        """
        # Set defaults
        defaults = {
            'cuda': torch.cuda.is_available(),
            'emb_dim': 300,
            'src_lang': 'zh',
            'tgt_lang': 'en',
            'normalize_embeddings': 'center,renorm',
            'dico_method': 'csls_knn_10',
            'dico_build': 'S2T',
            'dico_max_rank': 10000,
            'dico_max_size': 0,
            'dico_min_size': 0,
            'dico_threshold': 0,
            'map_id_init': True,
            'exp_path': './dumped',
            'exp_name': 'debug',
            'exp_id': '',
            'max_vocab': 200000,
        }

        # Update with user-provided values
        defaults.update(kwargs)

        # Set all as attributes
        for key, value in defaults.items():
            setattr(self, key, value)

    def __repr__(self):
        """String representation of params."""
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"SimpleParams({attrs})"


class OrthogonalMapper:
    """
    Learn orthogonal transformation matrix using Procrustes algorithm.

    This class wraps MUSE's Procrustes solution to align source embeddings
    to target embeddings using an orthogonal transformation matrix W.

    The optimization problem solved is:
        min ||WX - Y||² subject to W^T W = I

    Solution uses SVD: W = UV^T where UΣV^T = SVD(Y^T X)

    Examples:
        >>> # Load aligned word pairs
        >>> X_zh = zh_loader.get_embedding_matrix(zh_words)[0]  # (5000, 300)
        >>> X_en = en_loader.get_embedding_matrix(en_words)[0]  # (5000, 300)
        >>>
        >>> # Train mapper
        >>> mapper = OrthogonalMapper(normalize=True, center=True)
        >>> mapper.fit(X_zh, X_en)
        >>>
        >>> # Transform new Chinese words
        >>> X_zh_test = zh_loader.get_embedding_matrix(['猫', '狗'])[0]
        >>> X_aligned = mapper.transform(X_zh_test)
        >>>
        >>> # Save/load
        >>> mapper.save_mapping('mapping.npy')
        >>> mapper2 = OrthogonalMapper().load_mapping('mapping.npy')
    """

    def __init__(self, normalize: bool = True, center: bool = True):
        """
        Initialize the mapper.

        Args:
            normalize: If True, L2 normalize embeddings before alignment
            center: If True, center embeddings before alignment

        Examples:
            >>> # Full preprocessing (recommended)
            >>> mapper = OrthogonalMapper(normalize=True, center=True)
            >>>
            >>> # No preprocessing (use pre-normalized embeddings)
            >>> mapper = OrthogonalMapper(normalize=False, center=False)
        """
        self.normalize = normalize
        self.center = center
        self.W: Optional[np.ndarray] = None  # Transformation matrix
        self.mean_src: Optional[np.ndarray] = None  # Source mean (if centered)
        self.mean_tgt: Optional[np.ndarray] = None  # Target mean (if centered)

    def fit(
        self,
        X_source: np.ndarray,
        Y_target: np.ndarray
    ) -> 'OrthogonalMapper':
        """
        Learn orthogonal mapping from source to target embeddings.

        Uses MUSE's Procrustes algorithm from src/trainer.py:92-102.

        Args:
            X_source: Source embeddings (n_pairs, emb_dim)
            Y_target: Target embeddings (n_pairs, emb_dim)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If X and Y have different shapes

        Examples:
            >>> X = np.random.randn(5000, 300)
            >>> Y = np.random.randn(5000, 300)
            >>> mapper = OrthogonalMapper()
            >>> mapper.fit(X, Y)
            >>> print(mapper.W.shape)  # (300, 300)
        """
        from MUSE.src.trainer import Trainer
        from torch import nn

        if X_source.shape != Y_target.shape:
            raise ValueError(
                f"X_source and Y_target must have same shape. "
                f"Got X_source: {X_source.shape}, Y_target: {Y_target.shape}"
            )

        n_pairs, emb_dim = X_source.shape

        # Preprocess embeddings
        if self.center and self.normalize:
            X_proc = VectorPreprocessor.center_and_normalize(X_source)
            Y_proc = VectorPreprocessor.center_and_normalize(Y_target)
            self.mean_src = X_source.mean(axis=0)
            self.mean_tgt = Y_target.mean(axis=0)
        elif self.center:
            X_proc = VectorPreprocessor.center(X_source)
            Y_proc = VectorPreprocessor.center(Y_target)
            self.mean_src = X_source.mean(axis=0)
            self.mean_tgt = Y_target.mean(axis=0)
        elif self.normalize:
            X_proc = VectorPreprocessor.normalize(X_source)
            Y_proc = VectorPreprocessor.normalize(Y_target)
        else:
            X_proc = X_source.copy()
            Y_proc = Y_target.copy()

        # Convert to torch tensors
        X_torch = torch.from_numpy(X_proc).float()
        Y_torch = torch.from_numpy(Y_proc).float()

        # Create MUSE-compatible params
        params = SimpleParams(
            emb_dim=emb_dim,
            cuda=False,  # Use CPU for simplicity
            normalize_embeddings='',  # Already preprocessed
        )

        # Create nn.Embedding objects for MUSE
        src_emb = nn.Embedding(n_pairs, emb_dim, sparse=True)
        src_emb.weight.data.copy_(X_torch)
        tgt_emb = nn.Embedding(n_pairs, emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(Y_torch)

        # Create mapping layer (initialized to identity)
        mapping = nn.Linear(emb_dim, emb_dim, bias=False)
        mapping.weight.data.copy_(torch.eye(emb_dim))

        # Create dictionary indices (all pairs aligned 0->0, 1->1, etc.)
        dico = torch.from_numpy(np.column_stack([
            np.arange(n_pairs),
            np.arange(n_pairs)
        ])).long()

        # Add required attributes to params for Trainer
        # (Trainer expects these but we don't actually use them)
        params.src_dico = None
        params.tgt_dico = None

        # Create MUSE Trainer
        trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
        trainer.dico = dico

        # Apply Procrustes algorithm
        trainer.procrustes()

        # Extract learned transformation matrix
        self.W = mapping.weight.data.cpu().numpy()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned mapping to source embeddings.

        Args:
            X: Source embeddings to transform (n_samples, emb_dim)

        Returns:
            Aligned embeddings (n_samples, emb_dim)

        Raises:
            RuntimeError: If mapper has not been fitted yet

        Examples:
            >>> mapper = OrthogonalMapper()
            >>> mapper.fit(X_train, Y_train)
            >>> X_aligned = mapper.transform(X_test)
        """
        if self.W is None:
            raise RuntimeError("Mapper not fitted. Call fit() first.")

        # Apply same preprocessing as training
        if self.center and self.mean_src is not None:
            X_proc = X - self.mean_src
            if self.normalize:
                X_proc = VectorPreprocessor.normalize(X_proc)
        elif self.normalize:
            X_proc = VectorPreprocessor.normalize(X)
        else:
            X_proc = X

        # Apply transformation: X @ W^T
        return X_proc @ self.W.T

    def fit_transform(
        self,
        X_source: np.ndarray,
        Y_target: np.ndarray
    ) -> np.ndarray:
        """
        Fit mapper and transform source embeddings in one step.

        Args:
            X_source: Source embeddings (n_pairs, emb_dim)
            Y_target: Target embeddings (n_pairs, emb_dim)

        Returns:
            Aligned source embeddings (n_pairs, emb_dim)

        Examples:
            >>> X_aligned = OrthogonalMapper().fit_transform(X, Y)
        """
        self.fit(X_source, Y_target)
        return self.transform(X_source)

    def save_mapping(self, filepath: str) -> None:
        """
        Save transformation matrix and preprocessing params to file.

        Args:
            filepath: Path to save mapping (.npz format)

        Raises:
            RuntimeError: If mapper has not been fitted yet

        Examples:
            >>> mapper.fit(X, Y)
            >>> mapper.save_mapping('mapping.npz')
        """
        if self.W is None:
            raise RuntimeError("Mapper not fitted. Call fit() first.")

        np.savez(
            filepath,
            W=self.W,
            normalize=self.normalize,
            center=self.center,
            mean_src=self.mean_src if self.mean_src is not None else np.array([]),
            mean_tgt=self.mean_tgt if self.mean_tgt is not None else np.array([])
        )

    def load_mapping(self, filepath: str) -> 'OrthogonalMapper':
        """
        Load transformation matrix and preprocessing params from file.

        Args:
            filepath: Path to load mapping from (.npz format)

        Returns:
            self (for method chaining)

        Examples:
            >>> mapper = OrthogonalMapper()
            >>> mapper.load_mapping('mapping.npz')
            >>> X_aligned = mapper.transform(X_test)
        """
        data = np.load(filepath)
        self.W = data['W']
        self.normalize = bool(data['normalize'])
        self.center = bool(data['center'])

        mean_src = data['mean_src']
        self.mean_src = mean_src if mean_src.size > 0 else None

        mean_tgt = data['mean_tgt']
        self.mean_tgt = mean_tgt if mean_tgt.size > 0 else None

        return self


class Evaluator:
    """
    Evaluator for cross-lingual embeddings using mean cosine similarity.

    This class wraps MUSE's dist_mean_cosine() method for model selection
    during iterative refinement. It builds a temporary dictionary from
    aligned embeddings and computes the mean cosine similarity.

    Examples:
        >>> # Evaluate alignment quality
        >>> evaluator = Evaluator()
        >>> score = evaluator.mean_cosine_similarity(
        ...     X_src=X_zh,
        ...     Y_tgt=X_en,
        ...     W=mapper.W,
        ...     method='csls_knn_10'
        ... )
        >>> print(f"Mean cosine: {score:.4f}")
    """

    @staticmethod
    def mean_cosine_similarity(
        X_src: np.ndarray,
        Y_tgt: np.ndarray,
        W: np.ndarray,
        method: str = 'csls_knn_10',
        dico_max_size: int = 10000
    ) -> float:
        """
        Compute mean cosine similarity for model selection.

        Wraps MUSE's dist_mean_cosine() from src/evaluation/evaluator.py:35-68.

        This builds a temporary dictionary from aligned embeddings using
        CSLS or nearest neighbors, then computes the mean cosine similarity
        of the aligned pairs. Used for validation during iterative refinement.

        Args:
            X_src: Source embeddings (n_samples, emb_dim)
            Y_tgt: Target embeddings (n_samples, emb_dim)
            W: Transformation matrix (emb_dim, emb_dim)
            method: Dictionary building method - 'nn' or 'csls_knn_K' (default: 'csls_knn_10')
            dico_max_size: Maximum dictionary size for evaluation (default: 10000)

        Returns:
            Mean cosine similarity score (higher is better)

        Examples:
            >>> # After fitting mapper
            >>> score = Evaluator.mean_cosine_similarity(
            ...     X_src=X_zh,
            ...     Y_tgt=X_en,
            ...     W=mapper.W
            ... )
            >>> print(f"Validation score: {score:.4f}")
        """
        from MUSE.src.dico_builder import build_dictionary, get_candidates
        from torch import nn

        n_src, emb_dim = X_src.shape
        n_tgt = Y_tgt.shape[0]

        # Convert to torch
        X_torch = torch.from_numpy(X_src).float()
        Y_torch = torch.from_numpy(Y_tgt).float()
        W_torch = torch.from_numpy(W).float()

        # Create params for dictionary building
        params = SimpleParams(
            cuda=False,
            emb_dim=emb_dim,
            dico_method=method,
            dico_build='S2T',
            dico_threshold=0,
            dico_max_rank=10000,
            dico_min_size=0,
            dico_max_size=dico_max_size
        )

        # Create embeddings
        src_emb = nn.Embedding(n_src, emb_dim, sparse=True)
        src_emb.weight.data.copy_(X_torch)
        tgt_emb = nn.Embedding(n_tgt, emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(Y_torch)

        # Create mapping
        mapping = nn.Linear(emb_dim, emb_dim, bias=False)
        mapping.weight.data.copy_(W_torch)

        # Get aligned source embeddings
        src_aligned = mapping(src_emb.weight).data
        tgt_data = tgt_emb.weight.data

        # Normalize
        src_aligned = src_aligned / src_aligned.norm(2, 1, keepdim=True).expand_as(src_aligned)
        tgt_data = tgt_data / tgt_data.norm(2, 1, keepdim=True).expand_as(tgt_data)

        # Build dictionary
        s2t_candidates = get_candidates(src_aligned, tgt_data, params)
        t2s_candidates = get_candidates(tgt_data, src_aligned, params)
        dico = build_dictionary(src_aligned, tgt_data, params, s2t_candidates, t2s_candidates)

        # Compute mean cosine
        if dico is None or dico.size(0) == 0:
            return -1e9

        max_size = min(dico_max_size, dico.size(0))
        mean_cosine = (src_aligned[dico[:max_size, 0]] * tgt_data[dico[:max_size, 1]]).sum(1).mean()

        return float(mean_cosine.item())


class IterativeRefinementMapper(OrthogonalMapper):
    """
    Iteratively refine alignment by expanding dictionary.

    This class extends OrthogonalMapper to implement MUSE's iterative
    refinement procedure from supervised.py:78-103. It alternates between:
    1. Building/expanding dictionary from current alignment (using CSLS)
    2. Applying Procrustes with the expanded dictionary
    3. Evaluating and selecting the best iteration

    The refinement loop can significantly improve alignment quality by
    discovering more word pairs beyond the seed dictionary.

    Examples:
        >>> # Load seed dictionary pairs
        >>> X_zh = zh_loader.get_embedding_matrix(zh_words)[0]
        >>> X_en = en_loader.get_embedding_matrix(en_words)[0]
        >>>
        >>> # Train with refinement
        >>> mapper = IterativeRefinementMapper(n_refinement=5)
        >>> mapper.fit(X_zh, X_en)
        >>>
        >>> # Transform new words
        >>> X_aligned = mapper.transform(X_zh_test)
    """

    def __init__(
        self,
        n_refinement: int = 5,
        dico_max_rank: int = 10000,
        dico_method: str = 'csls_knn_10',
        dico_build: str = 'S2T',
        normalize: bool = True,
        center: bool = True
    ):
        """
        Initialize iterative refinement mapper.

        Args:
            n_refinement: Number of refinement iterations (default: 5)
            dico_max_rank: Max rank for dictionary candidates (default: 10000)
            dico_method: Method for finding candidates - 'nn' or 'csls_knn_K' (default: 'csls_knn_10')
            dico_build: Dictionary direction - 'S2T', 'T2S', or 'S2T&T2S' (default: 'S2T')
            normalize: L2 normalize embeddings (default: True)
            center: Center embeddings (default: True)

        Examples:
            >>> # Standard refinement
            >>> mapper = IterativeRefinementMapper(n_refinement=5)
            >>>
            >>> # More aggressive refinement with bidirectional dictionary
            >>> mapper = IterativeRefinementMapper(
            ...     n_refinement=10,
            ...     dico_method='csls_knn_10',
            ...     dico_build='S2T&T2S'
            ... )
        """
        super().__init__(normalize=normalize, center=center)
        self.n_refinement = n_refinement
        self.dico_max_rank = dico_max_rank
        self.dico_method = dico_method
        self.dico_build = dico_build
        self.best_score = -1e12
        self.iteration_scores = []

    def fit(
        self,
        X_source: np.ndarray,
        Y_target: np.ndarray,
        seed_dict_indices: Optional[np.ndarray] = None
    ) -> 'IterativeRefinementMapper':
        """
        Learn mapping with iterative refinement.

        Uses MUSE's refinement loop pattern from supervised.py:78-103:
        - Iteration 0: Uses seed dictionary (or all pairs if None)
        - Iteration 1+: Expands dictionary using current alignment

        Args:
            X_source: Source embeddings (n_samples, emb_dim)
            Y_target: Target embeddings (n_samples, emb_dim)
            seed_dict_indices: Initial dictionary as (n_seed, 2) array of indices.
                             If None, assumes all pairs aligned (0->0, 1->1, etc.)

        Returns:
            self (for method chaining)

        Examples:
            >>> # Use all pairs as seed (assumes perfect alignment)
            >>> mapper = IterativeRefinementMapper(n_refinement=5)
            >>> mapper.fit(X_zh, X_en)
            >>>
            >>> # Use subset as seed dictionary
            >>> seed_indices = np.column_stack([np.arange(1000), np.arange(1000)])
            >>> mapper.fit(X_zh, X_en, seed_dict_indices=seed_indices)
        """
        from MUSE.src.trainer import Trainer
        from MUSE.src.dico_builder import build_dictionary
        from torch import nn

        if X_source.shape != Y_target.shape:
            raise ValueError(
                f"X_source and Y_target must have same shape. "
                f"Got X_source: {X_source.shape}, Y_target: {Y_target.shape}"
            )

        n_samples, emb_dim = X_source.shape

        # Preprocess embeddings (same as OrthogonalMapper)
        if self.center and self.normalize:
            X_proc = VectorPreprocessor.center_and_normalize(X_source)
            Y_proc = VectorPreprocessor.center_and_normalize(Y_target)
            self.mean_src = X_source.mean(axis=0)
            self.mean_tgt = Y_target.mean(axis=0)
        elif self.center:
            X_proc = VectorPreprocessor.center(X_source)
            Y_proc = VectorPreprocessor.center(Y_target)
            self.mean_src = X_source.mean(axis=0)
            self.mean_tgt = Y_target.mean(axis=0)
        elif self.normalize:
            X_proc = VectorPreprocessor.normalize(X_source)
            Y_proc = VectorPreprocessor.normalize(Y_target)
        else:
            X_proc = X_source.copy()
            Y_proc = Y_target.copy()

        # Convert to torch
        X_torch = torch.from_numpy(X_proc).float()
        Y_torch = torch.from_numpy(Y_proc).float()

        # Create params
        params = SimpleParams(
            emb_dim=emb_dim,
            cuda=False,
            normalize_embeddings='',  # Already preprocessed
            dico_method=self.dico_method,
            dico_build=self.dico_build,
            dico_max_rank=self.dico_max_rank,
            dico_max_size=0,  # No limit
            dico_min_size=0,
            dico_threshold=0,
            exp_path='./dumped',
            exp_name='iterative_refinement',
            exp_id=''
        )

        # Create embeddings
        src_emb = nn.Embedding(n_samples, emb_dim, sparse=True)
        src_emb.weight.data.copy_(X_torch)
        tgt_emb = nn.Embedding(n_samples, emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(Y_torch)

        # Create mapping
        mapping = nn.Linear(emb_dim, emb_dim, bias=False)
        mapping.weight.data.copy_(torch.eye(emb_dim))

        # Add required attributes to params for Trainer
        params.src_dico = None
        params.tgt_dico = None

        # Create trainer
        trainer = Trainer(src_emb, tgt_emb, mapping, None, params)

        # Set seed dictionary
        if seed_dict_indices is None:
            # Use all pairs (assume aligned)
            seed_dict_indices = np.column_stack([np.arange(n_samples), np.arange(n_samples)])

        trainer.dico = torch.from_numpy(seed_dict_indices).long()

        # Create temp directory for saving best model
        os.makedirs(params.exp_path, exist_ok=True)
        exp_full_path = os.path.join(params.exp_path, params.exp_name)
        os.makedirs(exp_full_path, exist_ok=True)
        params.exp_path = exp_full_path

        # Refinement loop (following supervised.py:78-103)
        self.iteration_scores = []
        self.best_score = -1e12
        best_W = None

        for n_iter in range(self.n_refinement + 1):
            # Build dictionary from current alignment (skip iteration 0)
            if n_iter > 0:
                trainer.build_dictionary()

            # Apply Procrustes
            trainer.procrustes()

            # Get current W
            current_W = mapping.weight.data.cpu().numpy()

            # Evaluate using mean cosine
            score = Evaluator.mean_cosine_similarity(
                X_src=X_proc,
                Y_tgt=Y_proc,
                W=current_W,
                method=self.dico_method
            )

            self.iteration_scores.append(score)

            # Save best
            if score > self.best_score:
                self.best_score = score
                best_W = current_W.copy()
                print(f"Iteration {n_iter}: New best score = {score:.5f}")
            else:
                print(f"Iteration {n_iter}: Score = {score:.5f} (best = {self.best_score:.5f})")

        # Use best W
        if best_W is not None:
            self.W = best_W
        else:
            self.W = mapping.weight.data.cpu().numpy()

        return self


class VectorMapper:
    """
    High-level API for cross-lingual word embedding alignment.

    This class orchestrates all components (EmbeddingLoader, DictionaryParser,
    OrthogonalMapper/IterativeRefinementMapper) to provide a simple interface
    for aligning Chinese and English embeddings.

    Examples:
        >>> # Initialize
        >>> mapper = VectorMapper(
        ...     zh_embedding_path='wiki.zh.vec',
        ...     en_embedding_path='wiki.en.vec',
        ...     dict_path='cedict.u8'
        ... )
        >>>
        >>> # Load and train
        >>> mapper.load_resources()
        >>> mapper.train(method='iterative', n_refinement=5)
        >>>
        >>> # Use the mapping
        >>> translations = mapper.translate('猫', source_lang='zh', k=5)
        >>> print(translations)  # [('cat', 0.85), ('kitten', 0.72), ...]
    """

    def __init__(
        self,
        zh_embedding_path: str,
        en_embedding_path: str,
        dict_path: str,
        dict_format: str = 'cedict'
    ):
        """
        Initialize VectorMapper with file paths.

        Args:
            zh_embedding_path: Path to Chinese embeddings (Word2Vec format)
            en_embedding_path: Path to English embeddings (Word2Vec format)
            dict_path: Path to bilingual dictionary
            dict_format: Dictionary format - 'cedict' or 'muse' (default: 'cedict')

        Examples:
            >>> mapper = VectorMapper(
            ...     zh_embedding_path='wiki.zh.vec',
            ...     en_embedding_path='wiki.en.vec',
            ...     dict_path='cedict.u8'
            ... )
        """
        self.zh_embedding_path = zh_embedding_path
        self.en_embedding_path = en_embedding_path
        self.dict_path = dict_path
        self.dict_format = dict_format

        # Will be initialized by load_resources()
        self.zh_loader: Optional[EmbeddingLoader] = None
        self.en_loader: Optional[EmbeddingLoader] = None
        self.dict_parser: Optional[DictionaryParser] = None
        self.mapper: Optional[Union[OrthogonalMapper, IterativeRefinementMapper]] = None

        # Aligned data (set by load_resources())
        self.zh_words: Optional[List[str]] = None
        self.en_words: Optional[List[str]] = None
        self.X_zh: Optional[np.ndarray] = None
        self.X_en: Optional[np.ndarray] = None

    def load_resources(
        self,
        include_traditional: bool = False,
        max_vocab: Optional[int] = None
    ) -> 'VectorMapper':
        """
        Load embeddings and parse dictionary.

        Steps:
        1. Load Chinese and English embeddings
        2. Parse bilingual dictionary
        3. Filter dictionary to words with embeddings
        4. Create aligned embedding matrices

        Args:
            include_traditional: Include traditional Chinese variants (default: False)
            max_vocab: Maximum vocabulary size for embeddings (default: None = all)

        Returns:
            self (for method chaining)

        Examples:
            >>> mapper = VectorMapper(...)
            >>> mapper.load_resources()
            >>> print(f"Loaded {len(mapper.zh_words)} aligned pairs")
        """
        print("Loading Chinese embeddings...")
        self.zh_loader = EmbeddingLoader()
        self.zh_loader.load_word2vec(self.zh_embedding_path, max_vocab=max_vocab)
        zh_vocab = self.zh_loader.get_vocabulary()
        print(f"  Loaded {len(zh_vocab)} Chinese words")

        print("Loading English embeddings...")
        self.en_loader = EmbeddingLoader()
        self.en_loader.load_word2vec(self.en_embedding_path, max_vocab=max_vocab)
        en_vocab = self.en_loader.get_vocabulary()
        print(f"  Loaded {len(en_vocab)} English words")

        print(f"Parsing dictionary from {self.dict_path}...")
        self.dict_parser = DictionaryParser()
        if self.dict_format == 'cedict':
            self.dict_parser.parse_cedict(self.dict_path, include_traditional=include_traditional)
        elif self.dict_format == 'muse':
            self.dict_parser.parse_muse_format(self.dict_path, include_traditional=include_traditional)
        else:
            raise ValueError(f"Unknown dict_format: {self.dict_format}")

        print(f"  Parsed {len(self.dict_parser.get_pairs())} total pairs")

        print("Filtering dictionary by available embeddings...")
        self.dict_parser.filter_by_vocabulary(zh_vocab, en_vocab)
        pairs = self.dict_parser.get_pairs()
        print(f"  {len(pairs)} pairs with embeddings")

        if len(pairs) == 0:
            raise RuntimeError("No dictionary pairs with embeddings found!")

        # Extract word lists
        self.zh_words = [zh for zh, en in pairs]
        self.en_words = [en for zh, en in pairs]

        print("Creating aligned embedding matrices...")
        self.X_zh, zh_valid = self.zh_loader.get_embedding_matrix(self.zh_words, strict=True)
        self.X_en, en_valid = self.en_loader.get_embedding_matrix(self.en_words, strict=True)

        assert len(zh_valid) == len(en_valid), "Alignment mismatch!"
        print(f"  Created matrices: {self.X_zh.shape}")

        return self

    def train(
        self,
        method: str = 'iterative',
        n_refinement: int = 10,
        **kwargs
    ) -> 'VectorMapper':
        """
        Train the alignment mapper.

        Args:
            method: Training method - 'procrustes' or 'iterative' (default: 'procrustes')
            n_refinement: Number of refinement iterations (only for 'iterative')
            **kwargs: Additional arguments passed to mapper constructor

        Returns:
            self (for method chaining)

        Examples:
            >>> # Simple Procrustes
            >>> mapper.train(method='procrustes')
            >>>
            >>> # Iterative refinement
            >>> mapper.train(method='iterative', n_refinement=5)
        """
        if self.X_zh is None or self.X_en is None:
            raise RuntimeError("Resources not loaded. Call load_resources() first.")

        print(f"\nTraining with method: {method}")

        if method == 'procrustes':
            self.mapper = OrthogonalMapper(**kwargs)
            print("Applying Procrustes alignment...")
            self.mapper.fit(self.X_zh, self.X_en)
            print("Training complete!")

        elif method == 'iterative':
            self.mapper = IterativeRefinementMapper(n_refinement=n_refinement, **kwargs)
            print(f"Starting iterative refinement ({n_refinement} iterations)...")
            self.mapper.fit(self.X_zh, self.X_en)
            print(f"\nTraining complete! Best score: {self.mapper.best_score:.5f}")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'procrustes' or 'iterative'")

        return self

    def translate(
        self,
        word: str,
        source_lang: str,
        k: int = 5,
        use_csls: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Translate a word using nearest neighbor search.

        Args:
            word: Word to translate
            source_lang: Source language - 'zh' or 'en'
            k: Number of translations to return (default: 5)
            use_csls: Use CSLS instead of cosine similarity (default: False)

        Returns:
            List of (translation, score) tuples, sorted by score (descending)

        Examples:
            >>> translations = mapper.translate('猫', source_lang='zh', k=5)
            >>> for word, score in translations:
            ...     print(f"{word}: {score:.3f}")
        """
        if self.mapper is None:
            raise RuntimeError("Mapper not trained. Call train() first.")

        if source_lang == 'zh':
            if word not in self.zh_loader:
                raise KeyError(f"Word '{word}' not found in Chinese embeddings")

            # Get source vector and transform
            src_vec = self.zh_loader.get_embedding(word).reshape(1, -1)
            aligned_vec = self.mapper.transform(src_vec).flatten()

            # Get all target embeddings
            tgt_emb_dict = self.en_loader.get_embeddings()

        elif source_lang == 'en':
            if word not in self.en_loader:
                raise KeyError(f"Word '{word}' not found in English embeddings")

            # Get source vector (already aligned)
            src_vec = self.en_loader.get_embedding(word).reshape(1, -1)
            aligned_vec = src_vec.flatten()

            # Get all target embeddings (need to inverse transform)
            # For now, just use cosine similarity in original space
            tgt_emb_dict = self.zh_loader.get_embeddings()

        else:
            raise ValueError(f"Unknown source_lang: {source_lang}. Use 'zh' or 'en'")

        # Compute similarities
        if use_csls:
            # TODO: Implement CSLS properly
            # For now, fall back to cosine
            print("Warning: CSLS not fully implemented yet, using cosine similarity")
            use_csls = False

        if not use_csls:
            # Simple cosine similarity
            tgt_words = list(tgt_emb_dict.keys())
            tgt_vecs = np.vstack([tgt_emb_dict[w] for w in tgt_words])

            # Normalize
            aligned_vec_norm = aligned_vec / (np.linalg.norm(aligned_vec) + 1e-8)
            tgt_vecs_norm = tgt_vecs / (np.linalg.norm(tgt_vecs, axis=1, keepdims=True) + 1e-8)

            # Compute cosine similarities
            scores = tgt_vecs_norm @ aligned_vec_norm

            # Get top k
            top_k_indices = np.argsort(scores)[::-1][:k]
            results = [(tgt_words[i], float(scores[i])) for i in top_k_indices]

            return results

    def get_aligned_vector(self, word: str, source_lang: str) -> np.ndarray:
        """
        Get aligned embedding vector for a word.

        Args:
            word: Word to get embedding for
            source_lang: Source language - 'zh' or 'en'

        Returns:
            Aligned embedding vector

        Examples:
            >>> vec = mapper.get_aligned_vector('猫', source_lang='zh')
            >>> vec.shape  # (300,)
        """
        if self.mapper is None:
            raise RuntimeError("Mapper not trained. Call train() first.")

        if source_lang == 'zh':
            if word not in self.zh_loader:
                raise KeyError(f"Word '{word}' not found in Chinese embeddings")
            src_vec = self.zh_loader.get_embedding(word).reshape(1, -1)
            return self.mapper.transform(src_vec).flatten()

        elif source_lang == 'en':
            if word not in self.en_loader:
                raise KeyError(f"Word '{word}' not found in English embeddings")
            return self.en_loader.get_embedding(word)

        else:
            raise ValueError(f"Unknown source_lang: {source_lang}")

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model (.npz format)

        Examples:
            >>> mapper.save_model('alignment_model.npz')
        """
        if self.mapper is None:
            raise RuntimeError("Mapper not trained. Call train() first.")

        self.mapper.save_mapping(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> 'VectorMapper':
        """
        Load trained model from file.

        Args:
            filepath: Path to load model from (.npz format)

        Returns:
            self (for method chaining)

        Examples:
            >>> mapper = VectorMapper(...)
            >>> mapper.load_resources()
            >>> mapper.load_model('alignment_model.npz')
        """
        if self.X_zh is None:
            raise RuntimeError("Resources not loaded. Call load_resources() first.")

        self.mapper = OrthogonalMapper()
        self.mapper.load_mapping(filepath)
        print(f"Model loaded from {filepath}")

        return self