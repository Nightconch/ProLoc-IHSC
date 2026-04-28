# 🧬 ProLoc-IHSC
ProLoc-IHSC is a robust deep learning framework designed for highly accurate protein subcellular localization (SCL) prediction from immunohistochemistry (IHC) images.

Unlike conventional methods that assume conditional label independence, ProLoc-IHSC explicitly models the intrinsic biological dependencies among distinct subcellular compartments. By integrating label correlation learning with sequence-assisted features, it produces predictions that strictly align with empirical biological co-occurrence priors.

## Features
- Multi-modal fusion of image features (ViT) and protein sequence features (ProtT5)
- Cross-attention mechanism for effective feature integration
- Label relationship modeling using co-occurrence matrix
- Enhanced criterion learning strategy strategy
- Support for 5 subcellular locations: cytoplasm, endoplasmic reticulum, mitochondria, nucleus, plasma membrane

## Requirements

**Python Environment**

    python==3.11.7

**Install Dependencies**

    pip install -r requirements.txt

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
├── train.py              # Training script with label relationship loss
├── test.py               # Testing and evaluation script
├── model.py              # Cross-attention model architecture
├── prott5.py             # ProtT5 sequence feature extraction
├── vit.py                # ViT image feature extraction
├── metrics.py            # Evaluation metrics (sample-based & location-based)
├── requirements.txt      # Python dependencies
├── dataset/              # Dataset directory
│   ├── train/           # Training images
│   └── test/            # Testing images
├── embedding/           # Pre-extracted features
└── prot5/               # ProtT5 model files
```

## Quick Start
### 1.Create a Virtual Environment
To run the code, we need to create a virtual environment using Anaconda, and install the required dependencies. The command is as follows：
```
git git@github.com:Nightconch/ProLoc-IHSC.git
conda create -n ProLoc-IHSC pyhton=3.11.7
conda activate ProLoc-IHSC
pip install -r requirements.txt
```
### 2.Download pretrained model 
We use pre-trained Prott5, so you need to download the model and put it in the same directory as `train.py`.

Prott5: https://github.com/agemagician/ProtTrans   

model:ProtT5-XL-UniRef50 (also ProtT5-XL-U50)


### 3.Prepare your data
Proteins IHC images and sequences are necessary to perform ProLoc-IHS. IHC images should be of `.jpg` format, and sequences should be of `.csv` format. You can refer to the format in `dataset/test.csv` as a sequence example.

Attention: your IHC images and sequences should be in same order, or your will get wrong results.

### 4. Prepare Dataset

Organize your data in the following structure:

```
dataset/
├── train.csv            # Training data with columns: image_name, cytoplasm, endoplasmic_reticulum, mitochondria, nucleus, plasma_membrane, protein_sequence
├── test.csv             # Testing data with same format
├── train/               # Training IHC images
└── test/                # Testing IHC images
```

CSV format example:
```
image_name,cytoplasm,endoplasmic_reticulum,mitochondria,nucleus,plasma_membrane,protein_sequence
protein1.jpg,1,0,1,0,0,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

### 5. Test Model

```python
python test.py
```

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
