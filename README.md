<h2 align='center'>gcFINDR: gated continuous Flow-field Inference from Neural Data using Deep Recurrent Networks</h2>

This repository contains a modified version of [FINDR](https://github.com/Brody-Lab/findr) adapted for **continuous sEEG recordings** rather than spike trains. The core architecture — separating task-relevant dynamics from noise and learning a governing differential equation — is preserved, but the observation model has been replaced to accommodate log high-gamma power envelopes extracted from multi-site sEEG.

## Key modifications

| Component | Original FINDR | This version |
|---|---|---|
| Input signal | Spike counts (integer) | Log high-gamma power (continuous, float) |
| Observation model | Poisson likelihood | Gaussian likelihood |
| Per-channel noise | — | Learned variance σ² per electrode |
| Reconstruction loss | *x* log λ − λ | (*x* − μ)² / 2σ² + log σ |

### Details

**Gaussian observation model.** The Poisson model assumes non-negative integer observations with variance equal to the mean, which is inappropriate for log-transformed power signals. The replacement Gaussian likelihood treats each time-bin observation as a noisy readout of the latent state, with additive Gaussian noise of learned variance.

**Per-channel variance (σ²).** Signal-to-noise ratio varies substantially across sEEG electrode sites. A scalar σ² is learned independently for each channel during training, allowing the model to down-weight noisy electrodes without discarding them.

**Reconstruction loss.** The training objective replaces the Poisson log-likelihood with the Gaussian negative log-likelihood:

```
L_recon = Σ_i [ (x_i − μ_i)² / (2σ_i²) + log σ_i ]
```

where the sum is over channels *i*, *x_i* is the observed log high-gamma power, μ_i is the model's reconstructed output, and σ_i is the learned per-channel standard deviation.

---

## Background

Neural populations coordinate across large groups to perform tasks. FINDR's central premise is that the brain's underlying algorithm can be expressed as a differential equation over population activity. The method:

1. Separates task-relevant brain activity from task-irrelevant variability.
2. Learns the most likely differential equation consistent with the task-relevant dynamics.

This version extends that framework to the continuous-valued, multi-region sEEG signals commonly recorded during human intracranial studies.

---

## Installation

```bash
git clone https://github.com/laurenostrowski/MR-FINDR.git
module load anaconda/2024.10
conda create --name findr python=3.12
conda activate findr
cd findr
pip install -e .
```

---

## Data format

Data must be stored as an `.npz` file with the following fields:

| Field | Shape | Description |
|---|---|---|
| `spikes` | trials × time × channels | Log high-gamma power per time bin |
| `externalinputs` | trials × time × input_dim | Stimulus or task inputs (float or int) |
| `lengths` | trials | Trial duration in time bins |
| `times` | trials | Trial onset timestamps |

> **Note:** The `spikes` field is not renamed from the original FINDR, but represents log high-gamma power in this instantiation. Values should be log-transformed power envelopes (e.g. log of the analytic amplitude in the 70–150 Hz band) and z-scored (or otherwise normalized) across the session prior to input.

---

## Training

```bash
module load anaconda/2024.10
conda activate findr
python main.py --datapath=$datafilepath --workdir=$analysispath
```

`$datafilepath` — path to the `.npz` data file  
`$analysispath` — directory where trained model parameters are saved

Training typically takes a few hours on a single A100 GPU.

---

## Citation

If you use this modified version, please also cite the original FINDR paper:

Kim, T.D., Luo, T.Z., Can, T., Krishnamurthy, K., Pillow, J.W., Brody, C.D. (2025). Flow-field inference from neural data using deep recurrent networks. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*.

```bibtex
@article{kim2025findr,
    author    = {Timothy Doyeon Kim and Thomas Zhihao Luo and Tankut Can and
                 Kamesh Krishnamurthy and Jonathan W. Pillow and Carlos D. Brody},
    title     = {Flow-field inference from neural data using deep recurrent networks},
    year      = {2025},
    journal   = {Proceedings of the 42nd International Conference on Machine Learning (ICML)}
}
```
