import torch

with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()
print("Length of text:",len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

ctoi = { ch:i for i,ch in enumerate(chars) }
itoc = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: "".join([itoc[i] for i in l])

# split train/valid
n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8
n_embd = 32
device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_batch(split):
  data = train_data if split == "train" else valid_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  return x,y
xb, yb = get_batch(split="train")

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

def cal_loss(model):
  pass

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd),
                             nn.ReLU(),
                             nn.Linear(4* n_embd, n_embd)) ## for residual connections???
  
  def forward(self, x):
    return self.net(x)

class BigramLanguageModel(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(Block(n_embd, n_heads=4),
                                Block(n_embd, n_heads=4),
                                Block(n_embd, n_heads=4),
                                nn.LayerNorm(n_embd))
    self.dropout = nn.Dropout()
    self.lm_head = nn.Linear(n_embd, vocab_size)
  
  def forward(self, idx, targets=None):
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # (B, T, C) 
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb
    x = self.blocks(x)  # (B,T,C)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T,)
      #print(logits.shape, targets.shape)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx = (B, T)
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :] # (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx

class MultiHeadAttention(nn.Module):
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
  def forward(self, x):
    output = torch.cat([h[x] for h in self.heads], dim=-1)
    return self.proj(output)

class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa_head = MultiHeadAttention(n_embd, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    self.dropout = nn.Dropout()
  
  def foward(self, x):
    B,T,C = x.shape
    k = self.key(x) # B,T,C
    q = self.query(x) # B,T,C
    wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = self.dropout(F.softmax(wei, dim=-1)) # (B,T,T)

    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C) 
    return out


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape, loss)

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))