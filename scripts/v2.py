"""
GPT From Scratch — Version 2
Based on Andrej Karpathy's "Let's build GPT" lecture.

Stage: Minor code cleanup (58:26–1:00:18)

This version separates token embeddings from the language modeling
head. It is still a Bigram model, without self-attention.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# NOVO: número de características usadas para representar cada token.
# Antes, o embedding produzia diretamente 65 logits.
# Agora, cada token é representado por um vetor de 32 dimensões.
n_embd = 32

torch.manual_seed(1337)


# ------------------------------------------------------------
# Dataset and character-level tokenization
# ------------------------------------------------------------

# Executar o script a partir da raiz do repositório,
# onde o arquivo input.txt deve estar localizado.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Identifica os caracteres únicos do dataset.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Dicionários para converter caracteres em IDs e vice-versa.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Converte o dataset completo em IDs numéricos.
data = torch.tensor(encode(text), dtype=torch.long)

# Separa 90% para treino e 10% para validação.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def get_batch(split):
    # Seleciona o conjunto de treino ou validação.
    data = train_data if split == 'train' else val_data

    # Sorteia os pontos iniciais das sequências.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Entradas: B sequências de T tokens.
    x = torch.stack([data[i:i+block_size] for i in ix])

    # Targets: os mesmos trechos deslocados um token à frente.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # Move os tensores para o mesmo dispositivo do modelo.
    x, y = x.to(device), y.to(device)

    return x, y


# ------------------------------------------------------------
# Loss estimation
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    # Avalia o modelo sem calcular gradientes.
    out = {}

    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        # Média das losses dos batches avaliados.
        out[split] = losses.mean()

    model.train()
    return out


# ------------------------------------------------------------
# Bigram Language Model — Version 2
# ------------------------------------------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ANTES:
        # nn.Embedding(vocab_size, vocab_size)
        #
        # AGORA:
        # Cada token recebe uma representação com n_embd dimensões.
        # Exemplo: (65, 32).
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # NOVO:
        # Converte cada vetor de 32 características em 65 logits.
        # Um logit para cada possível próximo caractere.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx possui formato (B, T).

        # Obtém a representação vetorial de cada token.
        # (B, T) -> (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)

        # Projeta cada representação para o tamanho do vocabulário.
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(tok_emb)

        # Durante a geração, não temos targets.
        if targets is None:
            loss = None

        else:
            # Reorganiza os logits para calcular Cross-Entropy.
            B, T, C = logits.shape

            # (B, T, vocab_size) -> (B*T, vocab_size)
            logits = logits.view(B*T, C)

            # (B, T) -> (B*T,)
            targets = targets.view(B*T)

            # Calcula a perda média das previsões.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        # Gera um novo token em cada iteração.
        for _ in range(max_new_tokens):

            # Calcula os logits para o contexto atual.
            logits, loss = self(idx)

            # Utiliza somente a previsão da última posição.
            # (B, T, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :]

            # Transforma scores em probabilidades.
            probs = F.softmax(logits, dim=-1)

            # Sorteia um token com base nas probabilidades.
            idx_next = torch.multinomial(probs, num_samples=1)

            # Adiciona o novo token à sequência.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ------------------------------------------------------------
# Model initialization
# ------------------------------------------------------------

# O construtor não precisa mais receber vocab_size,
# pois essa variável já está definida no script.
model = BigramLanguageModel()
m = model.to(device)

# Otimizador responsável por atualizar os parâmetros.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for iter in range(max_iters):

    # Avalia periodicamente o modelo nos dois conjuntos.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # Seleciona um batch de treinamento.
    xb, yb = get_batch('train')

    # Forward: calcula as previsões e a loss.
    logits, loss = model(xb, yb)

    # Limpa os gradientes anteriores.
    optimizer.zero_grad(set_to_none=True)

    # Backward: calcula os gradientes.
    loss.backward()

    # Atualiza os parâmetros do modelo.
    optimizer.step()


# ------------------------------------------------------------
# Text generation
# ------------------------------------------------------------

# Contexto inicial: um único token de ID 0.
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Gera 500 novos tokens e converte os IDs em texto.
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
