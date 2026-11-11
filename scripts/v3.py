import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#------------------------------------------------------------

torch.manual_seed(1337)

# ------------------------------------------------------------
# Dataset and tokenization
# ------------------------------------------------------------

# Executar a partir da raiz do projeto, onde está input.txt.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapeamentos entre caracteres e IDs.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Codifica o texto e separa treino e validação.
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def get_batch(split):
    data = train_data if split == 'train' else val_data

    # Sorteia os índices iniciais das sequências.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x contém os tokens de entrada.
    x = torch.stack([data[i:i+block_size] for i in ix])

    # y contém os próximos tokens corretos.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


# ------------------------------------------------------------
# Loss estimation
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    # Avalia sem construir o grafo de gradientes.
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


# ------------------------------------------------------------
# Single Self-Attention Head — NOVO
# ------------------------------------------------------------

class Head(nn.Module):
    """Uma cabeça de Self-Attention causal."""

    def __init__(self, head_size):
        super().__init__()

        # Projeções lineares treináveis e independentes.
        # Q: o que o token procura?
        # K: quais características ele oferece?
        # V: qual conteúdo será compartilhado?
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Máscara triangular causal.
        # register_buffer registra tril como parte do módulo,
        # mas não como um parâmetro treinável.
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x possui formato (B, T, C).
        B, T, C = x.shape

        # Projeta os tokens em Keys e Queries.
        # (B, T, C) -> (B, T, head_size)
        k = self.key(x)
        q = self.query(x)

        # Calcula os scores de compatibilidade entre tokens.
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #
        # Divide por sqrt(head_size) para controlar
        # a escala dos scores antes do Softmax.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # Impede que tokens acessem posições futuras.
        # [:T, :T] permite processar sequências menores
        # que o block_size máximo.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # Converte os scores em pesos normalizados.
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Calcula os Values.
        v = self.value(x)

        # Combina os Values com os pesos de atenção.
        # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    """Múltiplas cabeças de Self-Attention."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Cria várias cabeças de Self-Attention.
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        # Projeta a saída concatenada das cabeças.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatena a saída de cada cabeça.
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Projeta a saída concatenada.
        out = self.dropout(self.proj(out))

        return out

# FeedForward: processa individualmente as representações
# de cada posição após a comunicação feita pelo Self-Attention.
class FeedForward(nn.Module):
    """Rede neural feed-forward simples."""

    def __init__(self, n_embd):
        super().__init__()

        # Uma camada linear seguida de ReLU e outra linear.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Um bloco Transformer: Self-Attention seguido de FeedForward."""

    def __init__(self, n_embd, n_head):
        # n_embd: embendding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head

        # Uma cabeça de Self-Attention.
        self.sa = MultiHeadAttention(n_head, head_size)

        # FeedForward.
        self.ffwd = FeedForward(n_embd)

        #layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Aplica Self-Attention e depois FeedForward.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ------------------------------------------------------------
# Language Model — Version 3
# ------------------------------------------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Representação treinável de cada token.
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )

        # representação treinável da posição.
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # Converte as representações em logits.
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Identidade dos tokens: (B, T, C).
        tok_emb = self.token_embedding_table(idx)

        # Posição de cada token: (T, C).
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )

        # Combina identidade e posição.
        # Broadcasting: (B,T,C) + (T,C) -> (B,T,C)
        x = tok_emb + pos_emb

        # contextualiza os tokens usando Self-Attention.
        x = self.blocks(x)

        # Aplica a normalização da camada final.
        x = self.ln_f(x)

        # Produz scores para os próximos tokens.
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # NOVO: limita o contexto aos últimos block_size
            # tokens, porque os embeddings posicionais só
            # foram definidos para essas posições.
            idx_cond = idx[:, -block_size:]

            # Calcula as previsões usando o contexto limitado.
            logits, loss = self(idx_cond)

            # Seleciona a previsão da última posição.
            logits = logits[:, -1, :]

            # Converte scores em probabilidades.
            probs = F.softmax(logits, dim=-1)

            # Sorteia o próximo token.
            idx_next = torch.multinomial(
                probs, num_samples=1
            )

            # Acrescenta o token à sequência completa.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ------------------------------------------------------------
# Model initialization
# ------------------------------------------------------------

model = BigramLanguageModel()
m = model.to(device)

# Mostra o número de parâmetros treináveis.
print(
    sum(p.numel() for p in m.parameters()) / 1e6,
    'M parameters'
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ------------------------------------------------------------
# Text generation
# ------------------------------------------------------------

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = m.generate(context, max_new_tokens=500)

print(decode(generated[0].tolist()))
