# 🧬 ProLoc-IHSC

ProLoc-IHSC is a robust deep learning framework for highly accurate protein subcellular localization (SCL) prediction from immunohistochemistry (IHC) images and protein sequences.

Unlike conventional methods that assume conditional label independence, ProLoc-IHSC explicitly models the intrinsic biological dependencies among distinct subcellular compartments. By integrating label correlation learning with multi-modal features (ViT for images + ProtT5 for sequences), it produces predictions that strictly align with empirical biological co-occurrence priors.

## Features
- Multi-modal fusion of image features (ViT) and protein sequence features (ProtT5)
- Cross-attention mechanism for effective feature integration
- Label relationship modeling using co-occurrence matrix
- Enhanced criterion learning strategy strategy
- Support for 5 subcellular locations: cytoplasm, endoplasmic reticulum, mitochondria, nucleus, plasma membrane
  
## Requirements

**Python Environment**

```bash
python==3.11.7
```

**Install Dependencies**

```bash
pip install -r requirements.txt
```

**Key Dependencies**
- torch==2.9.0+cu128
- transformers==4.48.1
- scikit-learn==1.7.2
- biopython==1.85
- pandas==2.3.3
- numpy==2.3.3
- matplotlib==3.10.7

## Project Structure

```
ProLoc-IHC-GitVersion/
├── predict.py            # Prediction script (main entry)
├── model.py              # Cross-attention model architecture
├── prott5.py             # ProtT5 sequence feature extraction
├── vit.py                # ViT image feature extraction
├── metrics.py            # Evaluation metrics
├── train.py              # Training script (for model development)
├── test.py               # Testing script (for model evaluation)
├── requirements.txt      # Python dependencies
├── results/
│   └── best_model.pth   # Pre-trained model weights
├── predict/              # Example prediction data
│   ├── images/          # Sample IHC images
│   └── sequences/       # Sample protein sequences (.fasta)
├── prot5/               # ProtT5 model files
└── embedding/           # Pre-extracted features cache
```

## Quick Start

### 1. Environment Setup

Clone the repository and create a virtual environment:

```bash
git clone git@github.com:Nightconch/ProLoc-IHSC.git
cd ProLoc-IHSC
conda create -n ProLoc-IHSC python=3.11.7
conda activate ProLoc-IHSC
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

**ProtT5 Model** (Required for sequence feature extraction)

Download ProtT5-XL-UniRef50 from [ProtTrans](https://github.com/agemagician/ProtTrans) and place it in the `prot5/` directory.

**ProLoc-IHSC Model** (Pre-trained weights)

The trained model is located at `results/best_model.pth`.

## Prediction Workflow

### Input Requirements

**IHC Images**
- Format: `.jpg`, `.jpeg`, or `.png`
- Content: Immunohistochemistry staining images of proteins

**Protein Sequences**
- Format: `.fasta` files or plain text
- Content: Amino acid sequences (standard 20 amino acids)
- Special characters (U, Z, O, B) will be automatically replaced with X

### Prediction Modes

#### Mode 1: Single Sample Prediction

Predict subcellular localization for a single protein:

```bash
python predict.py \
    --ihc path/to/image.jpg \
    --sequence path/to/sequence.fasta \
    --output results/prediction.csv
```

**Example:**

```bash
python predict.py \
    --ihc predict/ENSG00000005020-15519_B_7_7-HPA005560-nucleoplasm;cytosol.jpg \
    --sequence predict/ENSG00000005020-15519_B_7_7-HPA005560-nucleoplasm;cytosol.fasta \
    --output results/single_prediction.csv
```


#### Mode 2: Batch Prediction from Folders

Predict multiple proteins with images and sequences in separate folders:

```bash
python predict.py \
    --ihc path/to/images/ \
    --sequence path/to/sequences/ \
    --output predictions.csv
```

**Folder Structure:**

```
predict/
├── images/
│   ├── protein1.jpg
│   ├── protein2.jpg
│   └── protein3.jpg
└── sequences/
    ├── protein1.fasta
    ├── protein2.fasta
    └── protein3.fasta
```

**Important:** Image and sequence files must have matching names (excluding extensions).

**Example:**

```bash
python predict.py \
    --ihc predict/images/ \
    --sequence predict/sequences/ \
    --output results/batch_predictions.csv
```

### Output Format

All prediction results are saved as CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `image` | Image filename |
| `sequence_file` | Sequence filename (folder mode only) |
| `pred_cytoplasm` | Prediction for cytoplasm (0 or 1) |
| `pred_endoplasmic_reticulum` | Prediction for endoplasmic reticulum (0 or 1) |
| `pred_mitochondria` | Prediction for mitochondria (0 or 1) |
| `pred_nucleus` | Prediction for nucleus (0 or 1) |
| `pred_plasma_membrane` | Prediction for plasma membrane (0 or 1) |


## Model Training & Testing

For model development and evaluation, training and testing scripts are provided:

**Training:**
```bash
python train.py
```

**Testing:**
```bash
python test.py
```

These scripts are primarily for model development. For inference on new data, use `predict.py` as described above.

## Citation

If you use ProLoc-IHSC in your research, please cite:

```
[Your paper citation here]
```

## Contact

For questions and feedback:
- Email: liuyun313@jlu.edu.cn

## License

[Add your license information here]
