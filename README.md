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
- [x] Positional embeddings
- [x] Scaled dot-product self-attention (Q, K, V)
- [x] Single-head causal self-attention integration
- [x] Multi-head attention
- [x] Feedforward layers and residual connections
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

### Version 3 — Single-Head Causal Self-Attention

**File:** `scripts/v3.py`

Introduces the first attention-based language model following
the single-head integration stage of Karpathy's lecture.

**Implemented components:**

- Learned token and positional embeddings
- Query, Key, and Value linear projections
- Scaled dot-product attention
- Causal masking using a registered buffer
- Context-aware next-token prediction
- Context cropping during autoregressive generation

The model can now aggregate information from previous tokens
instead of relying exclusively on the current token.

This is a single-head attention model, not yet a complete
Transformer architecture.

### Version 3 Results

Pending local execution and evaluation.
```

Adicione também à seção de execução:

```bash
python scripts/v3.py
```

**Não registre nenhuma loss da V3 no README antes de executar o treinamento.** Queremos continuar usando somente resultados que você realmente obteve.

---

## 5. Testar no VS Code

Antes do commit, execute na raiz do projeto, com `.venv` ativado:

```bash
python -m py_compile scripts/v3.py
python scripts/v3.py
```

Verifique se o modelo imprime a quantidade de parâmetros, apresenta as losses periódicas e termina gerando texto.

Comparamos depois a validation loss com a do `bigram.py` e da `v2.py`, se você tiver registrado essa última. A expectativa é que uma cabeça de atenção melhore as previsões contextuais, mas não precisamos assumir um resultado antes do experimento.

## 6. Segundo commit — VS Code

Quando o script executar corretamente, utilize:

<box border radius="lg" padding={3} gap={2}>
  **Commit da implementação**
  <row align="center" gap={2}>
    <box background="surface-secondary" radius="md" padding={3} flex="1">
      <text size="sm" weight="medium">`feat: integrate single-head causal self-attention`</text>
    </box>
    <button color="secondary" variant="outline" size="sm" onClick={()=>GenUI.copy("feat: integrate single-head causal self-attention")}>
      <icon name="copy" size="sm"/>
    </button>
  </row>
</box>

```bash
git status
git add scripts/v3.py README.md
git commit -m "feat: integrate single-head causal self-attention"
git push origin main
git status
```

Se o push for rejeitado por novos commits do Colab, sincronize com `git pull --rebase origin main` e tente novamente, resolvendo qualquer conflito antes de prosseguir.

Ao concluir, teremos três implementações preservadas no GitHub: o Bigram original, a versão com embeddings separados e o modelo com uma cabeça de Self-Attention.

**Nosso próximo tópico será [1:21:59 — Multi-Head Self-Attention](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4919s).** Ainda não implementei múltiplas cabeças no código acima de propósito: vamos seguir a sequência do Karpathy e entender uma etapa antes de avançar para a seguinte.


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