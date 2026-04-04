# From One-Hot to Semantic Encoding: Entity Embedding for Small and Heterogeneous Digital Humanities Datasets

This repository accompanies the paper "From One-Hot to Semantic Encoding: Entity Embedding for Small and Heterogeneous Digital Humanities Datasets". It implements a pipeline to compare symbolic (one-hot) and semantic (LLM-based) representations of categorical metadata for a digital literature dataset coming from two distinct databases.
The pipeline of this paper is inspired by the [ARISE framework](https://github.com/develop-yang/ARISE/tree/main).

## Repository structure

```
.
├── semantic_encoding_DH.py      # Main script
├── DL_dataset.csv              # Dataset (wide binary metadata)
├── DL_descriptions.JSON        # Semantic descriptions
├── prompt.md                   # LLM prompt template
└── README.md
```

## Usage
```bash
python semantic_encoding_DH.py \
  --csv DL_dataset.csv \
  --descriptions DL_descriptions.JSON \
  --outdir outputs
```
