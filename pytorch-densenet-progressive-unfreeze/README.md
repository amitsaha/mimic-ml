# Unfreezing layers for transfer learning for classification

## Overview

Using densenet-121 as an example, we will learn how to "unfreeze" layers to fine tune a deep learning network on your own dataset. 

Some code but mostly high level description of what is involved and why you want to do it.


## Transfer learning

Repurposing of taking a model trained on one task and and uing it for a different task

Two primary paradigms:

1. Feature extraction - Frozen transfer learning
2. Fine-tuning - unfreeze layers

### Fine tuning

## Densenet-121

### Progressive learning of more complex representations

## Demo

### Peeking under the hood

Source: `peek.py`

### Frozen transfer learning

Source: `frozen_tl.py`

### Unfreezing a few layers within a denseblock

Source: `unfreeze_few.py`


### Unfreezing a denseblock

Source: `unfreeze_denseblock.py`

## Summary

