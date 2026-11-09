# GPT From Scratch

Building a character-level GPT language model from scratch using Python and PyTorch, following Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

The goal is to understand how language models work internally by implementing tokenization, autoregressive text generation, neural network training, self-attention, and the Transformer architecture.

## Learning Progress

- [x] Character-level tokenization
- [x] Dataset encoding and train/validation split
- [x] Data batching
- [x] Bigram language model
- [x] Cross-entropy loss
- [x] Model training and optimization (Bigram)
- [x] Training and validation loss evaluation
- [x] Autoregressive text generation (trained Bigram)
- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Transformer blocks
- [ ] Complete GPT architecture
- [ ] GPT training and evaluation

## Project Structure

```text
GPT_from_scratch/
├── scripts/
│   └── bigram.py
├── gpt_from_scratch.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

- **Notebook:** Learning notes, explanations, and experiments following the lecture.
- **bigram.py:** Standalone implementation of the character-level Bigram Language Model.
- **requirements.txt:** Python dependencies required to run the project.

## Bigram Language Model

The first implemented model uses a trainable embedding lookup table to predict the next character based only on the current character.

The training process includes:

- Random batch sampling from Tiny Shakespeare
- Forward propagation and cross-entropy loss
- Backpropagation using PyTorch autograd
- Parameter optimization with AdamW
- Training and validation loss estimation
- Autoregressive text generation

### Training Results

The Bigram model was trained for 3,000 iterations using the Tiny Shakespeare dataset.

| Metric | Loss |
|---|---:|
| Initial training loss (step 0) | 4.7305 |
| Initial validation loss (step 0) | 4.7241 |
| Last reported training loss (step 2700) | 2.4738 |
| Last reported validation loss (step 2700) | 2.4911 |

**Observations:**

- Both training and validation loss decreased substantially.
- The model learned local character transition patterns.
- Generated text remains largely incoherent because the Bigram model cannot use longer contextual relationships.
- The next objective is to implement self-attention and progressively build a Transformer-based GPT.

## Technologies

- Python
- PyTorch
- NumPy
- Google Colab
- Visual Studio Code
- NVIDIA CUDA (GPU acceleration)

## Running the Project

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the Tiny Shakespeare dataset into the project root:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

Run the Bigram model from the project root:

```bash
python scripts/bigram.py
```

The script trains the model, evaluates the loss, and generates sample text. GPU acceleration is used when a compatible CUDA environment is available.

## References

- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Official lecture repository](https://github.com/karpathy/ng-video-lecture)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)

## Project Status

**Work in progress.**

Currently transitioning from the Bigram Language Model to the fundamentals of self-attention and Transformer architecture.