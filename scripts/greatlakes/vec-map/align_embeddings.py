#!/usr/bin/env python3
"""
Cross-Lingual Word Embedding Alignment Script
==============================================

Aligns Chinese word embeddings to English space using Procrustes analysis
and exports a combined Word2Vec binary file containing both languages.

Usage:
    python align_embeddings.py \
        --zh-embeddings ../../pretrained_word2vec/zh/sgns.merge.word.bz2 \
        --en-embeddings ../../pretrained_word2vec/en/GoogleNews-vectors-negative300.bin.gz \
        --dictionary ../../dictionaries/cedict_processed.txt \
        --output-dir ../../pretrained_word2vec \
        --max-vocab 50000

Author: SI650 Project Team
Date: 2025-10-22
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
import logging

# Add project root to path to import vector_mapping
# Script is at: scripts/greatlakes/vec-map/align_embeddings.py
# Project root is 3 levels up
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from vector_mapping import VectorMapper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Align Chinese word embeddings to English space using Procrustes analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--zh-embeddings',
        type=str,
        required=True,
        help='Path to Chinese word embeddings (Word2Vec format, can be .bz2 compressed)'
    )

    parser.add_argument(
        '--en-embeddings',
        type=str,
        required=True,
        help='Path to English word embeddings (Word2Vec format, can be .gz compressed)'
    )

    parser.add_argument(
        '--dictionary',
        type=str,
        required=True,
        help='Path to bilingual dictionary (CEDICT processed format)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory where to save the aligned embeddings'
    )

    parser.add_argument(
        '--max-vocab',
        type=int,
        default=None,
        help='Maximum vocabulary size to load (for testing). None = load all.'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='L2 normalize embeddings before alignment'
    )

    parser.add_argument(
        '--center',
        action='store_true',
        default=True,
        help='Center embeddings at origin before alignment'
    )

    return parser.parse_args()


def create_combined_embeddings(mapper, embedding_dim):
    """
    Create a single KeyedVectors object containing both aligned Chinese and original English.

    Args:
        mapper: Trained VectorMapper instance
        embedding_dim: Dimensionality of embeddings

    Returns:
        gensim.models.KeyedVectors with combined vocabulary
    """
    logger.info("Creating combined embedding vocabulary...")

    # Get all words and vectors
    combined_words = []
    combined_vectors = []

    # Get embeddings from loaders
    zh_embeddings = mapper.zh_loader.get_embeddings()
    en_embeddings = mapper.en_loader.get_embeddings()

    # Add aligned Chinese embeddings
    logger.info(f"  Adding {len(zh_embeddings)} aligned Chinese words...")
    for word in zh_embeddings:
        aligned_vec = mapper.get_aligned_vector(word, 'zh')
        combined_words.append(f"zh:{word}")  # Prefix to distinguish language
        combined_vectors.append(aligned_vec)

    # Add original English embeddings
    logger.info(f"  Adding {len(en_embeddings)} English words...")
    for word in en_embeddings:
        combined_words.append(f"en:{word}")  # Prefix to distinguish language
        combined_vectors.append(en_embeddings[word])

    # Create KeyedVectors
    logger.info(f"  Total vocabulary: {len(combined_words)} words")
    kv = KeyedVectors(vector_size=embedding_dim)
    kv.add_vectors(combined_words, combined_vectors)

    return kv


def save_word2vec_binary(kv, output_path):
    """
    Save KeyedVectors as gzip-compressed Word2Vec binary format.

    Args:
        kv: gensim.models.KeyedVectors instance
        output_path: Path where to save (should end in .bin.gz)
    """
    logger.info(f"Saving to Word2Vec binary format: {output_path}")

    # Save as binary format with gzip compression
    # Note: gensim's save_word2vec_format doesn't directly support .gz,
    # so we'll save uncompressed first then compress
    temp_path = output_path.replace('.gz', '')

    logger.info("  Writing uncompressed binary file...")
    kv.save_word2vec_format(temp_path, binary=True)

    logger.info("  Compressing with gzip...")
    import gzip
    import shutil

    with open(temp_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove temporary uncompressed file
    os.remove(temp_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")


def main():
    """Main execution function."""
    args = parse_args()

    # Print configuration
    logger.info("=" * 70)
    logger.info("CROSS-LINGUAL EMBEDDING ALIGNMENT")
    logger.info("=" * 70)
    logger.info(f"Chinese embeddings:  {args.zh_embeddings}")
    logger.info(f"English embeddings:  {args.en_embeddings}")
    logger.info(f"Dictionary:          {args.dictionary}")
    logger.info(f"Output directory:    {args.output_dir}")
    logger.info(f"Max vocabulary:      {args.max_vocab if args.max_vocab else 'All'}")
    logger.info(f"Preprocessing:       normalize={args.normalize}, center={args.center}")
    logger.info("=" * 70)

    # Validate input files exist
    for path, name in [(args.zh_embeddings, "Chinese embeddings"),
                       (args.en_embeddings, "English embeddings"),
                       (args.dictionary, "Dictionary")]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Initialize VectorMapper
        logger.info("\n[1/5] Initializing VectorMapper...")
        mapper = VectorMapper(
            zh_embedding_path=args.zh_embeddings,
            en_embedding_path=args.en_embeddings,
            dict_path=args.dictionary,
            dict_format='muse'  # Using MUSE format (tab-separated)
        )

        # Load resources
        logger.info("\n[2/5] Loading embeddings and dictionary...")
        mapper.load_resources(max_vocab=args.max_vocab)

        # Get embedding dimension
        embedding_dim = mapper.X_zh.shape[1]
        logger.info(f"  Embedding dimension: {embedding_dim}")
        logger.info(f"  Aligned training pairs: {len(mapper.zh_words)}")

        # Train alignment
        logger.info("\n[3/5] Training Procrustes alignment...")
        mapper.train(
            method='iterative',
            normalize=args.normalize,
            center=args.center
        )

        # Create combined embeddings
        logger.info("\n[4/5] Creating combined vocabulary...")
        combined_kv = create_combined_embeddings(mapper, embedding_dim)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_filename = f"zh-en-mapped-{timestamp}.bin.gz"
        output_path = os.path.join(args.output_dir, output_filename)

        # Save combined embeddings
        logger.info("\n[5/5] Saving combined embeddings...")
        save_word2vec_binary(combined_kv, output_path)

        # Success summary
        logger.info("\n" + "=" * 70)
        logger.info("ALIGNMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Output file:     {output_path}")
        logger.info(f"Total words:     {len(combined_kv)}")
        logger.info(f"Chinese words:   {len(mapper.zh_loader.get_embeddings())} (aligned)")
        logger.info(f"English words:   {len(mapper.en_loader.get_embeddings())}")
        logger.info(f"Training pairs:  {len(mapper.zh_words)}")
        logger.info("=" * 70)

        logger.info("\nTo load the aligned embeddings:")
        logger.info(f"  from gensim.models import KeyedVectors")
        logger.info(f"  kv = KeyedVectors.load_word2vec_format('{output_filename}', binary=True)")
        logger.info(f"  # Access Chinese word: kv['zh:猫']")
        logger.info(f"  # Access English word: kv['en:cat']")

        return 0

    except Exception as e:
        logger.error(f"\n{'=' * 70}")
        logger.error("ERROR DURING ALIGNMENT")
        logger.error(f"{'=' * 70}")
        logger.error(f"{type(e).__name__}: {str(e)}", exc_info=True)
        logger.error(f"{'=' * 70}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
