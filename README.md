# GPT From Scratch

A character-level GPT language model implemented from scratch using **Python and PyTorch**, following Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

This project documents the step-by-step implementation of an autoregressive Transformer, from character-level tokenization and a simple Bigram baseline to multi-head causal self-attention, Transformer blocks, and language model training.

The objective is to understand the underlying architecture, mathematics, and training process rather than relying on pretrained language models.

## Learning Progress

### Language Modeling Fundamentals

- [x] Character-level tokenization
- [x] Dataset encoding and train/validation split
- [x] Random batch generation
- [x] Bigram language model
- [x] Cross-entropy loss
- [x] Backpropagation with PyTorch autograd
- [x] AdamW optimization
- [x] Training and validation loss estimation
- [x] Autoregressive text generation

### Attention and Transformer Architecture

- [x] Context aggregation with loops and matrix multiplication
- [x] Causal masking and masked softmax
- [x] Query, Key, and Value projections
- [x] Scaled dot-product attention
- [x] Single-head causal self-attention
- [x] Multi-head self-attention
- [x] Token and positional embeddings
- [x] Feedforward neural networks
- [x] Residual connections
- [x] Layer normalization
- [x] Dropout regularization
- [x] Transformer block implementation
- [x] Stacked Transformer architecture
- [x] GPT training and generation pipeline
- [ ] Document final GPT training metrics

## Project Structure

```text
GPT_from_scratch/
├── scripts/
│   ├── bigram.py
│   ├── v2.py
│   └── v3.py
├── gpt_from_scratch.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The notebook contains learning notes, mathematical explanations, and experiments. The Python scripts preserve the evolution of the model throughout the lecture.

The local `.venv/` directory and Tiny Shakespeare `input.txt` are excluded from version control.

## Model Implementations

### Version 1 — Bigram Language Model

**File:** `scripts/bigram.py`

A baseline language model that predicts the next character using only the current character.

Features include a trainable token lookup table, cross-entropy loss, AdamW optimization, validation loss estimation, and autoregressive text generation.

This model establishes a baseline before introducing contextual attention.

### Version 2 — Embedding and Language Modeling Head

**File:** `scripts/v2.py`

Refactors the initial model by separating token representations from next-token prediction.

Introduces:

- Token embeddings with 32 dimensions
- A separate linear language modeling head
- An intermediate representation suitable for future architectural extensions

This version does not yet perform contextual token interactions.

### Version 3 — Transformer-Based GPT

**File:** `scripts/v3.py`

Implements the main components of a decoder-only, autoregressive Transformer.

The architecture includes:

- Character-level token and positional embeddings
- Multi-head causal self-attention
- Scaled Query-Key dot products
- Learned Value projections
- Feedforward networks with ReLU activation
- Residual connections
- Layer normalization
- Dropout regularization
- Stacked Transformer blocks
- Linear language modeling head

The model predicts the next character using representations built from the available context.

## GPT Architecture

The final implementation uses the following configuration:

| Hyperparameter | Value |
|---|---:|
| Batch size | 64 |
| Context length | 256 tokens |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Dimensions per head | 64 |
| Transformer blocks | 6 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Configured training iterations | 5,000 |

### Architecture Overview

```text
Input Text
    |
Character-Level Tokenization
    |
Token Embeddings + Positional Embeddings
    |
6 x Transformer Blocks
    |-- LayerNorm
    |-- Multi-Head Causal Self-Attention
    |-- Residual Connection
    |-- LayerNorm
    |-- FeedForward Network
    |-- Residual Connection
    |
Final LayerNorm
    |
Linear Language Modeling Head
    |
Next-Token Logits
    |
Autoregressive Text Generation
```

Each attention head independently computes relationships between tokens. Causal masking prevents a token from accessing future positions.

The feedforward layers process the representations at each position, while residual connections and layer normalization support optimization across multiple blocks.

## Dataset

The project uses the **Tiny Shakespeare** dataset, a collection of Shakespearean text commonly used for character-level language modeling experiments.

- Tokenization: character-level
- Training split: 90%
- Validation split: 10%
- Objective: next-character prediction

The dataset is downloaded locally and is not included in the repository.

## Experimental Results

### Bigram Baseline

The baseline Bigram model was trained for 3,000 iterations.

| Metric | Cross-Entropy Loss |
|---|---:|
| Initial training loss (step 0) | 4.7305 |
| Initial validation loss (step 0) | 4.7241 |
| Last reported training loss (step 2700) | 2.4738 |
| Last reported validation loss (step 2700) | 2.4911 |

These values were obtained from the local Bigram training run.

The model learned local character transition patterns, but its generated text remained largely incoherent because predictions depended only on the current character.

### Transformer-Based GPT — Final Results

The decoder-only Transformer was trained on Tiny Shakespeare
using the following configuration:

- Training iterations: 5,000
- Batch size: 64
- Context length: 256
- Embedding dimension: 384
- Attention heads: 6
- Transformer blocks: 6
- Learning rate: 0.0003
- Optimizer: AdamW
- Dropout: 0.2

#### Final Training Metrics

| Metric | Loss |
|---|---:|
| Training loss (step 4999) | 1.0763 |
| Validation loss (step 4999) | 1.4873 |

#### Comparison with the Bigram Baseline

| Model | Validation Loss |
|---|---:|
| Bigram Language Model | 2.4911 |
| GPT Transformer | 1.4873 |

The Transformer achieved approximately 40.3% lower
validation cross-entropy loss than the Bigram baseline.

This improvement demonstrates the benefit of contextual
information processing through causal self-attention
and stacked Transformer blocks.

The gap between training and validation losses suggests
that generalization should be monitored for overfitting.

## Technologies

- Python
- PyTorch
- NumPy
- Google Colab / Jupyter
- Visual Studio Code
- NVIDIA CUDA

## Setup and Execution

Clone the repository:

```bash
git clone https://github.com/Guilherme5G/GPT_from_scratch.git
cd GPT_from_scratch
```

Create and activate a Python virtual environment on Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install a CUDA-enabled version of PyTorch compatible with your system, following the [official installation instructions](https://pytorch.org/get-started/locally/).

Download Tiny Shakespeare into the repository root:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

Run the implementations:

```bash
# Original Bigram baseline
python scripts/bigram.py

# Embedding and language modeling head
python scripts/v2.py

# Transformer-based GPT
python scripts/v3.py
```

The scripts execute their configured training loops, estimate training and validation losses, and generate sample text.

**Note:** The Transformer training script does not currently save checkpoints. Running it again initializes a new model rather than resuming a previous training run.

## References and Attribution

This is an educational implementation following Andrej Karpathy's lecture and accompanying code. It documents my study process, code adaptations, and experiments.

- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Official Lecture Repository — ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Project Status

**Lecture implementation completed — final experimental evaluation pending.**

The progression from a Bigram baseline to a decoder-only Transformer has been implemented. The next step is to document the final model's observed training results and evaluate its generated text.

This implementation is an educational character-level language model, not a reproduction of the original pretrained GPT-2.