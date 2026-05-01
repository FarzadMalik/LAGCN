# Sig2Vec++

Signature verification with learned 1D-CNN embeddings, built on top of the
[SynSig2Vec](https://github.com/LaiSongxuan/SynSig2Vec) architecture and
evaluated on the [MSDS-ChS](https://github.com/HCIILAB/MSDS) dataset.

## Repository Structure

```
sig2vec_pp/
├── configs/               # YAML experiment configs
│   ├── config.py          # Config loader
│   └── sig2vec_baseline.yaml
├── data/                  # Data loading & preprocessing
│   ├── download_msds.py   # Dataset verification / preprocessing
│   ├── msds_dataset.py    # PyTorch datasets (online, offline, multimodal)
│   └── preprocessing.py   # Time-series normalisation, resampling
├── experiments/           # Training & evaluation scripts
│   ├── train_sig2vec.py   # Training loop
│   └── evaluate.py        # Evaluation (EER, FNMR@FMR)
├── models/                # Model definitions
│   ├── selective_pooling.py
│   └── sig2vec.py         # Sig2Vec 1D-CNN (exact re-implementation)
├── utils/                 # Metrics, logging
│   ├── logging.py
│   └── metrics.py         # EER, DET curve, FNMR@FMR
├── requirements.txt
├── REPRODUCTION_CHECKLIST.md
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r sig2vec_pp/requirements.txt

# 2. Obtain MSDS-ChS (requires application — see checklist)
#    Extract to data/MSDS/MSDS-ChS/

# 3. Verify dataset
python -m sig2vec_pp.data.download_msds --root data/MSDS/MSDS-ChS

# 4. Train baseline
python -m sig2vec_pp.experiments.train_sig2vec \
    --config sig2vec_pp/configs/sig2vec_baseline.yaml \
    --device 0

# 5. Evaluate
python -m sig2vec_pp.experiments.evaluate \
    --config sig2vec_pp/configs/sig2vec_baseline.yaml \
    --weights checkpoints/sig2vec_epoch0199.pt \
    --device 0
```

## Dataset: MSDS-ChS

MSDS-ChS is a large-scale Chinese signature dataset (NeurIPS 2022 Spotlight):

| Property | Value |
|---|---|
| Users | 402 |
| Genuine samples | 8 040 (20 per user across 2 sessions) |
| Skilled forgeries | 8 040 (20 per user across 2 sessions) |
| Modalities | Online (X, Y, P, T, U) + Offline (rendered images) |

The dataset requires an approved application from SCUT DLVC Lab.
See [REPRODUCTION_CHECKLIST.md](REPRODUCTION_CHECKLIST.md) for details.

## Model: Sig2Vec 1D-CNN

This is a faithful re-implementation of the 1D-CNN from:

> Lai S, Jin L, Zhu Y, et al. *SynSig2Vec: Forgery-free learning of dynamic
> signature representations by Sigma Lognormal-based synthesis and 1D CNN.*
> IEEE TPAMI, 2021.

Architecture:
- 3 convolutional blocks (64 → 128 → 256 channels) with SELU activations
- FPN-style upsample + skip connection between blocks 2 and 3
- Multi-head selective (attention) pooling → 1024-dim embedding
- AP-DLM loss + label-smoothed cross-entropy for training

## References

```bibtex
@article{lai2021synsig2vec,
  title={SynSig2Vec: Forgery-Free Learning of Dynamic Signature Representations
         by Sigma Lognormal-Based Synthesis and 1D CNN},
  author={Lai, Songxuan and Jin, Lianwen and Zhu, Yecheng and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}

@inproceedings{zhang2022msds,
  title={MSDS: A Large-Scale Chinese Signature and Token Digit String Dataset
         for Handwriting Verification},
  author={Zhang, Peirong and Jiang, Jiajia and Liu, Yuliang and Jin, Lianwen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
