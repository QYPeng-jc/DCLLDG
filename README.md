# Dual-view contrastive learning of local topology and disease-mediated global semantics for drug-gene interaction prediction (DCLLDG)

DCLLDG is a dual-view contrastive learning method for DGI prediction based on local topological structure and disease-mediated global semantics. Specifically, DCLLDG jointly learns drug and gene representations from a local structural view and a disease-mediated global semantic view. The local structural encoding module is based on a drug-gene bipartite network and employs LightGCN to capture the direct neighborhood structure and multi-hop topological dependencies formed by known drug-gene associations. The global semantic encoding module further introduces disease nodes to construct a drug-gene-disease heterogeneous network and aggregates disease-mediated higher-order semantic information based on meta-paths. To promote information interaction between the two views, DCLLDG adopts a layer-wise interactive fusion strategy, enabling local topological representations and global semantic representations to continuously exchange and collaboratively update across different propagation layers. Meanwhile, a cross-view InfoNCE contrastive learning constraint is introduced to keep the representations of the same node consistent across the local structural view and the global semantic view, thereby enhancing the discriminative capability of the fused representations. In addition, to increase the challenge of the training task, we design a disease-constrained hard negative sampling strategy that constructs more challenging negative samples from unobserved drug–gene pairs with similar disease semantic backgrounds. Finally, the fused drug and gene representations are used to predict potential DGIs.

This repository provides the code and data for DCLLDG.

## Repository structure

```
DCLLDG/
├── Code/                         # Source code
│   ├── Main.py                   # Train and evaluate
│   ├── Model_sparse.py           # Local LightGCN branch and global meta-path branch
│   ├── DataHandler.py            # Load graphs and the train/test split
│   ├── Params.py                 # Command-line parameters
│   ├── run_main_norefine.py      # Optional launcher for the same command
│   └── Utils/
│       ├── Utils.py              # Classification loss, InfoNCE, and helpers
│       └── TimeLogger.py
├── Data/
│   ├── CTD_DGI.csv               # Drug-gene interactions
│   ├── CTD_RDAs.csv              # Drug-disease associations
│   ├── CTD_GDAs.csv              # Gene-disease associations
│   └── transductive/
│       ├── train.csv             # Training split
│       └── test.csv              # Test split
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

We collected DGIs from the Comparative Toxicogenomics Database (CTD). This database mainly contains literature-supported biological interactions and associations among chemicals, genes, and diseases. After filtering and preprocessing, a total of 10,288 interactions between 425 drugs and 595 genes were obtained. To assist the model in learning disease-mediated semantics, we further extracted association information between diseases and drugs/genes from CTD, yielding 27,007 drug-disease associations (RDAs) and 9,861 gene-disease associations (GDAs), involving 595 genes.

| Drugs | Genes | Diseases | DGIs | RDAs | GDAs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 425 | 595 | 955 | 10,288 | 27,007 | 9,861 |

## Usage

Run all commands from the `Code/` directory.

```bash
python Main.py --iteration 5
```

## Requirements

```bash
pip install -r requirements.txt
```

Install a CUDA build of PyTorch that matches your GPU if the default wheel is CPU-only. Training uses `cuda:0` by default.
