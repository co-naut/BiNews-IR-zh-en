# Download
Download the data from [the official website](https://www.cs.jhu.edu/~shuosun/clirmatrix/).

You should download the following files:
- BI-139 en -> zh
- BI-139 zh -> en
- Documents zh
- Documents en

# Data processing

## Normalize Chinese characters
Run the following command to convert traditional Chinese characters in the documents to simplified Chinese.
```bash
python convert_document.py -i zh.tsv.gz -o zh_normalized.tsv.gz
```

## Prepare queries
Run the following command to convert the query file to a format expected by BM25 baseline.
```bash
python convert_query.py -i input.jl.gz -q queries.tsv -r qrels.txt
```

## Translate queries
Run the following command to translate queries.
```bash
export OPENAI_API_KEY=sk-...
python queries_translate_query.py \
  --input queries.tsv \
  --output queries_zh.tsv \
  --model gpt-4.1-mini
```