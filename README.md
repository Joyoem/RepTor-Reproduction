# RepTor Reproduction

Unofficial PyTorch reimplementation of:

> Eunik Park, Daehyun Ahn, Hyungjun Kim.  
> **RepTor: Re-parameterizable Temporal Convolution for Keyword Spotting via Differentiable Kernel Search.**  
> Interspeech 2024.

Paper: https://www.isca-archive.org/interspeech_2024/park24_interspeech.html

## Scope

This repository separates two goals:

1. **Faithful architecture reproduction**
   - RepTorBlock with multi-branch temporal convolutions
   - identity + BatchNorm branch when input/output shapes match
   - structural re-parameterization into one temporal convolution
   - RepTor-k and searched RepTor-A architectures

2. **Experimental reproduction**
   - GSC v2 preprocessing/training/evaluation
   - accuracy/MAC/latency reproduction
   - device deployment

The first goal is implemented here. The training pipeline is intentionally kept separate so that paper-faithful code is not mixed with later MSWC / Jetson adaptations.

## Paper architecture implemented

The paper uses 40-dimensional MFCC input. Treating MFCC bins as channels gives input shape:

```text
[B, 40, T]
```

Base channel sequence:

```text
[16, 24, 24, 32, 32, 48, 48]
```

Network:

```text
Conv1d(k=3, s=1, c=16)
RepTorBlock(k=?, s=2, c=24)
RepTorBlock(k=?, s=1, c=24)
RepTorBlock(k=?, s=2, c=32)
RepTorBlock(k=?, s=1, c=32)
RepTorBlock(k=?, s=2, c=48)
RepTorBlock(k=?, s=1, c=48)
GlobalAveragePooling
Dropout(0.2)
Linear
```

For **RepTor-9**, every RepTorBlock uses `kmax=9`.

For **RepTor-A**, Figure 5 of the paper gives:

```text
[7, 9, 5, 9, 5, 5]
```

## RepTorBlock

At training time, `RepTorBlock-kmax` contains all temporal convolution branches whose kernel sizes are <= `kmax`, selected from `{9, 7, 5, 3, 1}`. Each convolution branch is followed by BatchNorm. If stride=1 and input/output channels are equal, an additional identity BatchNorm branch is used.

At deployment time:

1. Fuse each Conv + BN.
2. Convert identity + BN into an equivalent convolution.
3. Zero-pad smaller kernels to `kmax`.
4. Sum all branch kernels and biases.
5. Replace the whole block with one Conv1d.

## Quick check

```bash
python scripts/verify_reparam.py
python scripts/inspect_model.py --model reptor9
python scripts/inspect_model.py --model reptor_a
```

The important first sanity checks are:

- output before/after re-parameterization should differ only by floating point error;
- deployed RepTor-9 should be approximately **62.1K parameters**;
- deployed RepTor-A should be approximately **42.9K parameters**.

The paper reports:

| Model | Accuracy | MACs | Params |
|---|---:|---:|---:|
| RepTor-9 | 97.30% | 1.51M | 62.1K |
| RepTor-A | 97.38% | 1.19M | 42.9K |

Exact MAC counting depends on profiler convention and input frame count.

## Recommended reproduction workflow

```text
Stage 1  core block implementation
Stage 2  re-parameterization equivalence test
Stage 3  parameter / MAC sanity check
Stage 4  GSC v2 paper-faithful training
Stage 5  reproduce reported accuracy
Stage 6  adapt to MSWC / project-specific intents
Stage 7  Jetson latency benchmark
```

## Status

- [x] RepTorBlock
- [x] Conv-BN fusion
- [x] Identity-BN fusion
- [x] RepTor-k model
- [x] RepTor-A architecture
- [x] deploy conversion
- [x] re-parameterization unit test
- [ ] exact GSC v2 data recipe
- [ ] paper augmentation recipe
- [ ] paper training recipe
- [ ] TFLite/mobile latency reproduction
- [ ] MSWC adaptation
- [ ] Jetson benchmark

## Disclaimer

This is an independent reimplementation based on the published paper and is not an official repository from the paper authors.
