# Sig2Vec++ Reproduction Checklist

This document tracks the steps needed to fully reproduce the Sig2Vec baseline
results on MSDS-ChS and prepare for Sig2Vec++ experiments.

---

## 1. Environment Setup

- [ ] Python 3.9+ installed
- [ ] Create virtual environment: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install dependencies: `pip install -r sig2vec_pp/requirements.txt`
- [ ] Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## 2. Dataset Acquisition (MSDS-ChS)

- [ ] Apply for MSDS dataset at https://github.com/HCIILAB/MSDS
- [ ] Download and complete the Application Form and Legal Commitment
- [ ] Submit via the SCUT DLVC Lab portal
- [ ] After approval, download and extract to `data/MSDS/MSDS-ChS/`
- [ ] Verify layout:
  ```
  data/MSDS/MSDS-ChS/
    session1/
      0/
        images/  g_0_0.png  f_0_0.png  …
        series/  g_0_0.txt  f_0_0.txt  …
      1/ …
      …
      401/ …
    session2/
      0/ … 401/ …
  ```
- [ ] Run verification: `python -m sig2vec_pp.data.download_msds --root data/MSDS/MSDS-ChS`

## 3. Data Preprocessing

- [ ] Compute global normalisation statistics:
  ```bash
  python -m sig2vec_pp.data.download_msds --root data/MSDS/MSDS-ChS --compute-stats --features x y p
  ```
- [ ] Confirm `data/MSDS/MSDS-ChS/global_stats.npz` was created

## 4. Train Sig2Vec Baseline

- [ ] Review config: `sig2vec_pp/configs/sig2vec_baseline.yaml`
- [ ] Launch training:
  ```bash
  python -m sig2vec_pp.experiments.train_sig2vec \
      --config sig2vec_pp/configs/sig2vec_baseline.yaml \
      --device 0
  ```
- [ ] Confirm checkpoints appear in `./checkpoints/`
- [ ] Training converges (mAP trending upward, AP loss decreasing)

## 5. Evaluate Baseline

- [ ] Run evaluation:
  ```bash
  python -m sig2vec_pp.experiments.evaluate \
      --config sig2vec_pp/configs/sig2vec_baseline.yaml \
      --weights checkpoints/sig2vec_epoch0199.pt \
      --device 0
  ```
- [ ] Record metrics:
  - EER: ______
  - FNMR@FMR=0.01: ______
  - FNMR@FMR=0.001: ______

## 6. Expected Results (Sig2Vec Baseline on MSDS-ChS)

| Metric           | Original Paper (DeepSignDB) | MSDS-ChS Target |
|------------------|-----------------------------|------------------|
| EER              | ~3–5% (varies by protocol)  | TBD              |
| FNMR@FMR=0.01   | –                           | TBD              |

> **Note:** The original Sig2Vec paper reports results on DeepSignDB, not
> MSDS-ChS.  Results on MSDS-ChS will serve as our own baseline for
> Sig2Vec++ improvements.

## 7. Key Hyperparameters to Verify

| Parameter          | Value   | Source                    |
|--------------------|---------|---------------------------|
| Optimizer          | SGD     | Original Sig2Vec          |
| Learning rate      | 0.001   | Original Sig2Vec          |
| Momentum           | 0.9     | Original Sig2Vec          |
| Weight decay       | 1e-5    | Original Sig2Vec          |
| Epochs             | 200     | Original Sig2Vec          |
| Tasks/batch        | 4       | Original Sig2Vec          |
| Genuine shots      | 5       | Original Sig2Vec          |
| Forgery shots      | 10      | Original Sig2Vec          |
| AP-DLM alpha       | 6.0     | Original Sig2Vec          |
| Label smoothing    | 0.1     | Original Sig2Vec          |
| LR schedule        | ×0.1 at epoch 150 | Original Sig2Vec |
| Input features     | X, Y, P | Adapted for MSDS-ChS      |

## 8. Differences from Original Sig2Vec

The original SynSig2Vec pipeline includes sigma-lognormal signature
synthesis for forgery-free training.  In this baseline:

- We use **real** genuine + skilled-forgery pairs from MSDS-ChS (no synthesis)
- The 1D-CNN architecture is kept **identical**
- The AP-DLM loss + label-smoothed CE loss are kept **identical**
- Input features are X, Y, P (pressure) — the original uses the same

## 9. Next Steps (Sig2Vec++)

- [ ] Integrate offline image branch (ResNet/ViT encoder)
- [ ] Multimodal fusion of online + offline embeddings
- [ ] Advanced augmentation strategies
- [ ] Explore transformer-based temporal encoders
- [ ] Benchmark on MSDS-TDS subset
