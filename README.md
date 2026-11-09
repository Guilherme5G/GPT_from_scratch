# GPT From Scratch

A hands-on implementation of a character-level language model using **Python and PyTorch**, following Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

The goal is to understand language models from first principles by progressively implementing tokenization, autoregressive training, causal attention, and the Transformer architecture.

This repository documents the learning process, intermediate implementations, and experimental results.

## Learning Progress

### Foundations

- [x] Character-level tokenization
- [x] Dataset encoding and train/validation split
- [x] Random data batching
- [x] Bigram language model
- [x] Cross-entropy loss
- [x] Backpropagation and AdamW optimization
- [x] Training and validation loss estimation
- [x] Autoregressive text generation

### Attention and Transformers

- [x] Causal context aggregation using loops
- [x] Matrix multiplication for weighted aggregation
- [x] Lower-triangular causal masking
- [x] Masked softmax and numerical equivalence checks
- [x] Embedding dimension and language-modeling head separation
- [ ] Positional embeddings
- [ ] Self-attention (Query, Key, Value)
- [ ] Multi-head attention
- [ ] Feedforward layers and residual connections
- [ ] Transformer blocks and layer normalization
- [ ] Complete GPT architecture
- [ ] GPT training and evaluation

## Project Structure

```text
GPT_from_scratch/
├── scripts/
│   ├── bigram.py
│   └── v2.py
├── gpt_from_scratch.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The local `.venv/` environment and `input.txt` dataset are excluded from version control.

## Implementations

### Version 1 — Bigram Language Model

**File:** `scripts/bigram.py`

A baseline character-level language model using a trainable embedding lookup table that directly produces next-token logits.

**Implemented concepts:**

- Character-level encoding and decoding
- Random batches of token sequences
- Cross-entropy loss
- AdamW optimization
- Train/validation evaluation
- Autoregressive generation

The model predicts the next character using only the current character.

### Version 2 — Embedding and Language Modeling Head

**File:** `scripts/v2.py`

A refactored Bigram model following the *Minor Code Cleanup* section of Karpathy's lecture.

Changes include:

- Introduction of `n_embd = 32`
- Token embeddings separated from next-token logits
- A linear language-modeling head (`lm_head`)
- Simplified model initialization

The new computation is:

```text
Token IDs
   ↓
Embedding(vocab_size, n_embd)
   ↓
Linear(n_embd, vocab_size)
   ↓
Next-token logits
```

This version remains a Bigram model. Contextual token interactions, positional embeddings, and self-attention are not yet implemented.

## Experimental Results

### Bigram Baseline

**Dataset:** Tiny Shakespeare  
**Training iterations:** 3,000  
**Optimizer:** AdamW  
**Batch size:** 32  
**Context length:** 8

| Metric | Loss |
|---|---:|
| Initial training loss (step 0) | 4.7305 |
| Initial validation loss (step 0) | 4.7241 |
| Last reported training loss (step 2700) | 2.4738 |
| Last reported validation loss (step 2700) | 2.4911 |

**Observations:**

Training and validation losses decreased substantially, indicating that the model learned local character transition patterns.

Generated samples remain largely incoherent because the baseline model cannot integrate information from earlier tokens beyond the current character.

The losses above refer specifically to `bigram.py`.

### Version 2 Results

Pending local execution and evaluation. Results will be recorded separately to avoid mixing measurements from different implementations.

## Technologies

- Python
- PyTorch
- NumPy
- Google Colab
- Visual Studio Code
- NVIDIA CUDA

## Setup and Execution

Clone the repository:

```bash
git clone https://github.com/Guilherme5G/GPT_from_scratch.git
cd GPT_from_scratch
```

Create and activate a virtual environment on Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For CUDA-enabled PyTorch installations, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Download Tiny Shakespeare into the project root:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

Run the baseline:

```bash
python scripts/bigram.py
```

Run Version 2:

```bash
python scripts/v2.py
```

Both scripts train their models, estimate training and validation losses, and generate sample text. They use CUDA when a compatible PyTorch environment and GPU are available.

## References and Attribution

This educational project follows and adapts the work presented by Andrej Karpathy. The implementations are based on his lecture and accompanying code, with additional learning notes, documentation, and experiments.

- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Official Lecture Repository (MIT License)](https://github.com/karpathy/ng-video-lecture)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)

## Project Status

**Work in progress — Foundations of Self-Attention.**

The Bigram baseline is implemented and trained. Version 2 introduces an intermediate embedding representation and a language-modeling head.

The next milestone is positional encoding, followed by the first implementation of causal self-attention using Query, Key, and Value projections.