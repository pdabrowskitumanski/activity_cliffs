# Activity Cliffs

A comprehensive tool for analyzing activity cliffs in molecular datasets. Activity cliffs are pairs of structurally similar molecules with large differences in biological activity — a phenomenon that poses challenges for structure-activity relationship (SAR) modeling.

## Installation

```bash
uv sync
```

> **macOS:** `activity_cliffs.py` defaults `OMP_NUM_THREADS` to `1`, because the OpenMP runtimes bundled with PyTorch, scikit-learn and XGBoost crash when run multithreaded in one process. Set `OMP_NUM_THREADS` yourself to override.

## Quick Start

All commands use configuration files stored in the `configs/` directory:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Calculate embeddings and dataset statistics (optional; speeds up later analyses)
python activity_cliffs.py embed sample_embedding

# Run the analysis pipeline
python activity_cliffs.py analyze sample_pipeline

# Compare results of several analysis runs
python activity_cliffs.py compare sample_comparison
```

`configs/sample_embedding.json`, `configs/sample_pipeline.json` and `configs/sample_comparison.json` are ready-to-run examples on the SARS-CoV-2 dataset. `sample_comparison` compares Morgan fingerprints with and without chirality, so it expects a second `analyze` run on the chiral embeddings (`"embeddings_file": "data/SARS_CoV2_embeddings_chiral.pkl"`, `"output_dir": "results/SARS_CoV2/morgan_tanimoto_chiral"`).

---

## Commands

The tool provides three main commands:

| Command | Description |
|---------|-------------|
| `analyze` | Run the full activity cliffs analysis pipeline |
| `embed` | Pre-compute embeddings and dataset statistics |
| `compare` | Compare results from multiple analysis runs |

---

## Datasets

Every config selects its data with the `"dataset"` key, which is either:

- **the name of a standard dataset** — downloaded on first use from the permalinks in `src/dataset_sources.json` and cached in `data/`:

  | Name | Activity | Source |
  |------|----------|--------|
  | `SARS_CoV2` | IC50 [µM] against SARS-CoV-2 main protease | PostEra COVID Moonshot |
  | `ChEMBL_Dopamine_D2` | Ki [nM] against dopamine D2 receptor | ChEMBL |
  | `ChEMBL_Factor_Xa` | Ki [nM] against factor Xa | ChEMBL |

  All three come as cleaned molecule and matched molecular pair (MMP) files from [Dablander et al., QSAR-activity-cliff-experiments](https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments). Column names, unit and censor threshold are taken from `src/dataset_sources.json`, and the MMPs are used automatically by `analyze`.

- **a path to a local CSV file** — with a SMILES column and an activity column (see [Data Format](#data-format)). Outputs are named after the file name without extension.

Dataset settings can be set or overridden in any config:

| Parameter | Description | Default (local CSV) |
|-----------|-------------|---------------------|
| `smiles_column` | Name of the SMILES column | `"smiles"` |
| `activity_column` | Name of the activity column | `"activity"` |
| `unit` | Activity unit: `"nM"`, `"uM"`, `"mM"`, `"M"` | `"nM"` |
| `censor_threshold` | Entries with `activity >= threshold` are marked as censored | `null` |
| `mmp_file` | CSV with matched molecular pairs (`smiles1`/`smiles2` or `smiles_1`/`smiles_2` columns) | `null` |

## Configuration Files

All commands are driven by JSON configuration files in the `configs/` directory.

### Analyze Config

Used with: `python activity_cliffs.py analyze <config_name>` (example: `configs/sample_pipeline.json`)

```json
{
    "dataset": "SARS_CoV2",
    "embeddings_file": "data/SARS_CoV2_embeddings.pkl",
    "output_dir": "results/SARS_CoV2/morgan_tanimoto",
    "embedding_method": "morgan",
    "distance_metric": "tanimoto",
    "n_compounds": null,
    "activity_threshold": 1.0,
    "similarity_threshold": null,
    "seed": 42
}
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `dataset` | ✅ | Standard dataset name or path to a CSV file (see [Datasets](#datasets)) | - |
| `embedding_method` | ✅ | Embedding method to use | - |
| `distance_metric` | ✅ | Distance metric to use | - |
| `embeddings_file` | ❌ | Pickle produced by `embed` to take embeddings from (see below) | - |
| `output_dir` | ❌ | Directory for the results | `results/<name>/<embedding>_<metric>_<hash>` |
| `n_compounds` | ❌ | Limit number of compounds (for testing). Use `null` for all. | `null` |
| `n_bits` | ❌ | Number of bits for the Morgan fingerprint* | `1024` |
| `radius` | ❌ | Radius for the Morgan fingerprint* | `2` |
| `chirality` | ❌ | Include stereochemistry in the Morgan fingerprint, so stereoisomers differ* | `false` |
| `activity_threshold` | ❌ | Log ratio threshold for activity cliff definition | `1.0` |
| `similarity_threshold` | ❌ | Similarity threshold for filtering pairs | `null` |
| `seed` | ❌ | Random seed for data splits and model training | `42` |

\* Used when `analyze` computes Morgan fingerprints itself. With `embeddings_file`, they are taken from the file and can be omitted; if given, they must agree with it.

**Where embeddings come from:**
- With `embeddings_file`, that pickle is used. It must contain `embedding_method`, and its fingerprint settings (recorded by `embed`) define `n_bits`, `radius` and `chirality`. To analyze chiral Morgan fingerprints, point it to a pickle made by `embed` with `"chirality": true`, e.g. `data/SARS_CoV2_embeddings_chiral.pkl`, and choose a matching `output_dir` such as `results/SARS_CoV2/morgan_tanimoto_chiral`.
- Without it, `analyze` reuses `data/<name>_embeddings.pkl` if that contains `embedding_method` with the same fingerprint settings, and otherwise computes the embeddings from the CSV.

#### Embedding Methods

| Method | Description |
|--------|-------------|
| `morgan` | Morgan circular fingerprint (optionally chiral) |
| `maccs` | MACCS 166-bit structural keys |
| `rdkit` | RDKit default fingerprint |
| `atom_pair` | Atom pair fingerprint |
| `topological_torsion` | Topological torsion fingerprint |
| `molformer` | MolFormer embedding (requires model download) |
| `chemberta` | ChemBERTa embedding, mean-pooled last hidden state (requires model download) |
| `chemeleon` | CheMeleon embedding (requires model download) |

#### Distance Metrics

| Metric | Description |
|--------|-------------|
| `tanimoto` | Tanimoto distance (1 - Tanimoto similarity) |
| `dice` | Dice distance (1 - Dice similarity) |
| `cosine` | Cosine distance (1 - cosine similarity) |
| `l2` / `euclidean` | Euclidean (L2) distance |
| `l1` / `manhattan` | Manhattan (L1) distance |
| `hamming` | Hamming distance |
| `sokal_michener` | Sokal-Michener distance |

### Embed Config

Used with: `python activity_cliffs.py embed <config_name>` (example: `configs/sample_embedding.json`)

```json
{
    "dataset": "SARS_CoV2",
    "methods": ["morgan", "maccs", "rdkit", "atom_pair", "topological_torsion", "molformer", "chemberta", "chemeleon"],
    "output_file": "data/SARS_CoV2_embeddings.pkl",
    "n_bits": 1024,
    "radius": 2,
    "chirality": false,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "save_csv": false
}
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `dataset` | ✅ | Standard dataset name or path to a CSV file (see [Datasets](#datasets)) | - |
| `methods` | ❌ | List of embedding methods to compute (see [Embedding Methods](#embedding-methods)). `null` or missing computes all. | all |
| `output_file` | ❌ | Path of the output `.pkl` file | `data/<name>_embeddings.pkl` |
| `n_bits` | ❌ | Number of bits for fingerprints | `1024` |
| `radius` | ❌ | Radius for Morgan fingerprint | `2` |
| `chirality` | ❌ | Include stereochemistry in the Morgan fingerprint | `false` |
| `umap_n_neighbors` / `umap_min_dist` | ❌ | UMAP projection parameters | `15` / `0.1` |
| `save_csv` | ❌ | Also save embeddings as CSV | `false` |

#### Activity Unit Conversion

The `unit` parameter determines how `log_activity` is calculated:

```
log_activity = -log10(activity) + shift
```

| Unit | Shift | Description |
|------|-------|-------------|
| `nM` | 9 | Nanomolar (default) |
| `uM` | 6 | Micromolar |
| `mM` | 3 | Millimolar |
| `M` | 0 | Molar |

This converts raw activity values to a pIC50-like scale where higher values indicate higher potency.

### Compare Config

Used with: `python activity_cliffs.py compare <config_name>` (example: `configs/sample_comparison.json`)

```json
{
    "name": "SARS_CoV2_morgan_chirality",
    "description": "Morgan fingerprints with and without chirality on the SARS-CoV-2 dataset",
    "dataset": "SARS_CoV2",
    "configs": [
        "results/SARS_CoV2/morgan_tanimoto",
        "results/SARS_CoV2/morgan_tanimoto_chiral"
    ]
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `dataset` | ✅ | Dataset the compared runs were made on (as in their analyze configs) |
| `configs` | ✅ | Results of `analyze` runs: their `output_dir` paths, or directory names under `results/<dataset>/` |
| `name` | ❌ | Name of the comparison; results go to `results/<dataset>/comparison/<name>/` |
| `description` | ❌ | Description included in the report |

---

## 1. Analyze

The `analyze` command runs the complete activity cliffs analysis pipeline on a molecular dataset.

### Usage

```bash
source .venv/bin/activate
python activity_cliffs.py analyze my_config
```

### Analysis Pipeline

The `analyze` command performs four main analysis steps:

#### Step 1: Dataset Analysis

Analyzes the molecular dataset and calculates pairwise distances.

**What it does:**
- Calculates vector length statistics for embeddings (with Gaussian fit)
- Computes UMAP projection of molecular embeddings
- Calculates center of embedding space and distances from center
- Calculates pairwise distances between all molecules
- Builds distance and similarity matrices
- Handles censored data by marking pairs as 'correct', 'censored', or 'excluded'

Note: Activity statistics are pre-computed during `embed` (Step 3) and loaded from file.

**Outputs (in `dataset_analysis/` subdirectory):**
- `vector_length_statistics.json` / `.png` — Embedding vector length statistics
- `umap_projection.csv` / `.png` — UMAP coordinates and visualization
- `center_of_space.json` — Center of embedding space
- `raw_distances.json` — All pairwise distances
- `distance_statistics.json` / `.png` — Distance statistics (2×2 panel)
- `distance_matrix.npy` / `similarity_matrix.npy` — Matrices
- `distance_similarity_matrices.png` — Matrix visualization (2×2 panel)
- `summary.json` — Complete analysis summary

#### Step 2: Cliff Identification

Identifies activity cliffs in the dataset.

**What it does:**
- Calculates activity differences as log ratio: `|log₁₀(activity₁) - log₁₀(activity₂)|`
- Identifies activity cliffs as pairs where `log_ratio ≥ activity_threshold`
- Calculates cliff fraction as a function of similarity threshold
- Fits sigmoid functions to activity cliff fraction curves

**Outputs (in `cliff_identification/` subdirectory):**
- `activity_cliff_fractions.csv` — Cliff fraction vs similarity threshold
- `activity_cliff_fractions.png` — Two-panel visualization
- `activity_cliff_sets.json` — Sets of cliff pairs at each threshold
- `fit_parameters.json` — Sigmoid fit parameters
- `summary.json` — Summary of cliff identification

#### Step 3: Correlation Analysis

Analyzes the correlation between structural similarity and activity differences.

**What it does:**
- Calculates regularized Lipschitz values: `log_ratio / max(1 - similarity, ε)`
- Builds and plots Lipschitz heatmap
- Fits Weibull decay to Lipschitz distribution
- Analyzes log-Lipschitz distribution by similarity bin (5×2 plot)
- Calculates Pearson correlation vs similarity bin
- Calculates cliff probability vs similarity bin

**Outputs (in `correlation/` subdirectory):**
- `lipschitz_heatmap.png` — Heatmap of Lipschitz values
- `lipschitz_matrix.npy` — Lipschitz value matrix
- `lipschitz_distribution.png` — Distribution with Weibull fit (2×2 panel)
- `binned_log_lipschitz.png` — Log-Lipschitz by similarity bin (5×2 panel)
- `binned_pearson_correlation.csv` / `.png` — Correlation vs similarity
- `binned_cliff_probability.csv` / `.png` — Cliff probability vs similarity
- `summary.json` — Summary of correlation analysis

#### Step 4: Persistent Homology Analysis

Applies topological data analysis to understand the structure of activity cliffs.

**What it does:**
- **Singly filtrated PH:** Builds a Vietoris-Rips complex filtered by log ratio values
- **Doubly filtrated PH:** Creates slices at different similarity thresholds and computes PH for each
- Computes persistence diagrams and statistics for dimensions 0, 1, and 2

**Outputs (in `persistent_homology/` subdirectory):**
- `singly_filtrated_dimX.csv` — Persistence diagrams per dimension
- `singly_filtrated_diagram.png` — Persistence and barcode diagrams
- `singly_filtrated_stats.json` — Statistics summary
- `doubly_filtrated_simX.X_dimY.csv` — Diagrams for each similarity slice
- `doubly_filtrated_simX.X_diagram.png` — Diagrams for each slice
- `doubly_filtrated_comparison.png` — How PH features change with threshold
- `doubly_filtrated_stats.json` — Statistics across all slices

#### Step 5: Model Training

Trains contrastive learning models to predict activity cliffs from molecular embeddings.

**What it does:**
- Splits data into train/validation/test sets (7:2:1 ratio)
- Trains neural network classifiers for different similarity thresholds
- Evaluates accuracy, precision, recall, and F1 score on test set

**Outputs (in `model_training/` subdirectory):**
- `training_results.json` — Full training configuration and metrics
- `accuracy_vs_threshold.csv` — Performance metrics for each threshold
- `accuracy_vs_threshold.png` — Visualization of metrics vs threshold

### Output Structure

All results are saved to `output_dir`, by default `results/<dataset>/<embedding>_<metric>_<hash>/`, where `<hash>` identifies the full analysis settings:

```
results/SARS_CoV2/morgan_tanimoto_1a2b3c4d/
├── REPORT.md                          # Comprehensive markdown report
├── dataset_analysis/
│   ├── summary.json
│   ├── vector_length_statistics.json
│   ├── vector_length_statistics.png
│   ├── activity_statistics.json
│   ├── activity_statistics.png
│   ├── umap_projection.csv
│   ├── umap_projection.png
│   ├── center_of_space.json
│   ├── raw_distances.json
│   ├── distance_statistics.json
│   ├── distance_statistics.png
│   ├── distance_matrix.npy
│   ├── similarity_matrix.npy
│   ├── activity_distance_matrix.npy
│   ├── activity_similarity_matrix.npy
│   ├── distance_similarity_matrices.png
│   └── scaling_info.json
├── cliff_identification/
│   ├── summary.json
│   ├── activity_cliff_fractions.csv
│   ├── activity_cliff_fractions.png
│   ├── activity_cliff_sets.json
│   └── fit_parameters.json
├── correlation/
│   ├── summary.json
│   ├── lipschitz_heatmap.png
│   ├── lipschitz_matrix.npy
│   ├── lipschitz_distribution.png
│   ├── binned_log_lipschitz.png
│   ├── binned_pearson_correlation.csv
│   ├── binned_pearson_correlation.png
│   ├── binned_cliff_probability.csv
│   └── binned_cliff_probability.png
├── persistent_homology/
│   ├── singly_filtrated_dim0.csv
│   ├── singly_filtrated_dim1.csv
│   ├── singly_filtrated_diagram.png
│   ├── singly_filtrated_stats.json
│   ├── doubly_filtrated_sim0.3_dim0.csv
│   ├── doubly_filtrated_sim0.3_diagram.png
│   ├── ... (for each threshold)
│   ├── doubly_filtrated_comparison.png
│   └── doubly_filtrated_stats.json
└── model_training/
    ├── training_results.json
    ├── accuracy_vs_threshold.csv
    └── accuracy_vs_threshold.png
```

---

## 2. Embed

The `embed` command pre-computes embeddings for a dataset (all available methods, or those listed in `methods`) and saves them as a pickle file. This speeds up subsequent analysis runs.

### Usage

```bash
source .venv/bin/activate
python activity_cliffs.py embed sample_embedding
```

### Embed Pipeline

The `embed` command performs these steps:

#### Step 1: Calculate Embeddings

Calculates the selected embeddings (all by default) for each molecule in the dataset.

**Computed Embeddings:**
- **Morgan fingerprint** — Circular fingerprint with configurable radius and size
- **MACCS keys** — 166-bit structural keys
- **RDKit fingerprint** — RDKit's default fingerprint
- **Atom pair fingerprint** — Based on atom pair descriptors
- **Topological torsion** — Based on topological torsion descriptors
- **MolFormer** — Transformer-based molecular embedding
- **ChemBERTa** — Transformer-based molecular embedding
- **CheMeleon** — Message-passing neural network embedding

#### Step 2: Process and Save

Processes the data and saves as pickle file.

**What it does:**
- Applies unit conversion to log_activity: `log_activity = -log10(activity) + shift`
- Marks censored entries based on `censor_threshold`
- Saves the processed DataFrame as a pickle file

**Output:**
- `output_file` (default `data/<name>_embeddings.pkl`) — Processed embeddings

#### Step 3: Activity Statistics

Calculates and plots activity statistics.

**What it does:**
- Calculates statistics for all compounds and uncensored compounds
- Fits Normal distribution to uncensored activities
- Generates 3-panel activity statistics plot

**Outputs (in `results/<name>/dataset_analysis/`):**
- `activity_statistics.json` — Activity statistics and Normal fit parameters
- `activity_statistics.png` — 3-panel plot (all activities, uncensored with fit, Q-Q plot)

---

## 3. Compare

The `compare` command compares results from multiple analysis runs. This is useful for comparing different embedding methods, similarity metrics, or datasets.

### Usage

```bash
source .venv/bin/activate
python activity_cliffs.py compare sample_comparison
```

*Note: This feature is under development. More comparison functionality will be added in future versions.*

---

## Modules

| Module | Description |
|--------|-------------|
| `src/embeddings.py` | Molecular embedding generation (Morgan, MACCS, RDKit, etc.) |
| `src/compound_similarity.py` | Vector similarity/distance metrics |
| `src/single_dataset_parser.py` | Main analysis pipeline |
| `src/datasets.py` | Dataset resolution: standard datasets (`src/dataset_sources.json`) or local CSV files |
| `src/cliff_identification.py` | Activity cliff identification and statistics |
| `src/correlation.py` | Correlation analysis and Lipschitz statistics |
| `src/persistent_homology.py` | Topological data analysis with ripser |
| `src/model_training.py` | Contrastive learning model for cliff prediction |
| `src/plotting.py` | Visualization utilities |
| `src/printing.py` | Markdown report generation |
| `src/comparison.py` | Multi-run comparison |
| `src/dataset_preparation.py` | Dataset preparation and embedding building |
| `src/chemistry.py` | Chemistry utilities (InChI key generation) |
| `src/utils.py` | General utilities (config loading) |

---

## Data Format

Local datasets should be CSV files with at least:
- A SMILES column containing molecular structures
- An activity column containing biological activity values (e.g., IC50, Ki)
- An optional `censored` column (boolean) to mark unreliable measurements

Example:
```csv
smiles,activity,censored
CCO,0.5,false
CCN,1.2,false
CCC,10.0,true
...
```

When the `censored` column is present:
- Pairs where both compounds are censored are marked as 'excluded' (set to NaN)
- Pairs where one compound is censored are marked as 'censored'
- Pairs where neither compound is censored are marked as 'correct'

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
