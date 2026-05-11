# SOLUTION — SMILES-2026 Signal Interference Cancellation

## 1. Reproducibility Instructions

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
scipy
tqdm
```

### Running

```bash
python3 applicant_solution.py
```

The script loads `challenge.mat`, runs the baseline and the proposed solution, prints per-channel reduction in dB, and writes `results.json`.

***

## 2. Expected Output

```
=== Baseline ===
  ch0: 3.98 dB
  ch1: 4.86 dB
  ch2: 3.49 dB
  ch3: 3.74 dB
  Metric [baseline]: 4.02 dB

=== Your Solution ===
=== Iteration 1 ===
=== Iteration 2 ===
=== End of fit ===
  ch0: 9.64 dB
  ch1: 7.91 dB
  ch2: 9.58 dB
  ch3: 7.21 dB
  Metric [yours]: 8.59 dB
```

| | ch0 | ch1 | ch2 | ch3 | Average |
|---|---|---|---|---|---|
| Baseline | 3.98 dB | 4.86 dB | 3.49 dB | 3.74 dB | 4.02 dB |
| **Proposed** | **9.64 dB** | **7.91 dB** | **9.58 dB** | **7.21 dB** | **8.59 dB** |

***

## 3. Solution Description

### Algorithm

The canceller performs two successive iterations, each consisting of a TX regression step followed by a rank-1 extraction step.

#### Feature Library

A library of 72 nonlinear TX candidates is constructed for all pairs `(i, j)` among the 6 TX channels:

```
tx[i]² · conj(tx[j])        — third-order intermodulation
tx[i] · |tx[j]|²            — gain compression cross-terms
```

Each candidate is bandpass-filtered with the scorer's filter. Lags from −10 to +10 samples are applied to each candidate, yielding 72 × 21 = 1512 complex features.

#### TX Regression (`fit_tx`)

A complex Ridge regression (`α = 0.01`) is fitted on the `MODEL_SUBSET` window [20 000 .. 220 000] where the signal is stationary:

Prediction on the full signal is computed in batches of 20 000 samples to control memory usage.

#### Iteration 1

1. Fit TX regression on `rx` → `tx_pred1`
2. Compute residual `r1 = rx − tx_pred1`
3. Extract rank-1 external component from the bandpass-filtered `r1` via eigendecomposition of the spatial covariance matrix → `rank1_1`
4. Clean signal: `rx − tx_pred1 − rank1_1`

#### Iteration 2

5. Fit TX regression on the cleaned signal → `tx_pred2`  
   The external interference `E` has been removed, so the regression estimates `F_c(TX)` more accurately.

#### Final Output

```
rx_clean = rx − tx_pred1 − rank1_1 − tx_pred2
```

Every subtracted component is by construction explainable by the scorer: `tx_pred1` and `tx_pred2` are linear combinations of TX-driven features; `rank1_1` is a rank-1 spatial component.


### What Did Not Work

- **Neural network (LSTM and CNN) approach**: a sequence model trained on windowed `(tx, rx)` pairs with a rank-1-based loss failed the explainability check. The model learned components that did not decompose into TX + rank-1 as required by the scorer, resulting in a forced score of 0 dB regardless of prediction quality. (explainability ~0.9)