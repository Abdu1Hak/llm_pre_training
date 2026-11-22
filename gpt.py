import torch
import torch.nn as nn 
from torch.nn import functional as F  

# hyper parameters 
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
# ------ 

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#####  ------------------- 1. TOKENIZER -----------------------
# all unique characters in the set 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping tokenizers 
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test split 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loader - from the train data randomly 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get 32 random integers as start tokens 
    x = torch.stack([data[i:i+block_size] for i in ix]) # retreive the next 8 tokens for each random 32 integers
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # get each consective token list 
    x, y = x.to(device), y.to(device)
    return x, y 

@torch.no_grad() 
def estimate_loss(): 
    out = {} 
    model.eval() # Interesting
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters): 
            X, Y = get_batch(split) 
            logits, loss = model(X, Y) 
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Interesting
    return out 

##### ------------- 2. MODEL (BIGRAM -> TRANSFORMERS) -----------------------

##### --------------- 2a. Self-Attention -----------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix

        self.dropout = nn.Dropout(dropout)
        # randomly drops some activations encouraging the model to learn new features when individual signals are missing
        # improves generalization and reduces overfitting (regularization technique)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)   # (B,T,C) -> (B,T,head_size)
        q = self.query(x) # (B,T,C) -> (B,T,head_size)
        # compute attention scores 
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C) -> (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

##### --------------- 2b. Multi-Head Self-Attention -----------------------

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # project back to original embedding dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1) # concatenate along the channel dimension
        out = self.proj(out) # project back to original embedding dimension
        return out


##### --------------- 2c. Feed-Forward Network -----------------------
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # growing the layer to 4 times the embedding dimension in the non-residual pathway
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer to project back to residual pathway
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.net(x)



#### ------------------- 2d. Residual Connections / Pre-Norm (Optimization # 1 + 2 - Deep NN) -----------------------

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, num_heads):
        # n_embd: embedding dimension, num_heads: number of heads in multi-head attention
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # residual connection 
        x = x +self.ffwd(self.ln2(x)) # residual connection 
        return x 


#### ------------------- 2e. Bigram Language Model -----------------------

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 64, 32
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 8, 32
        self.blocks = nn.Sequential(* [Block(n_embd, num_heads=n_head) for _ in range(n_layer)]) # 4 transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #  B T C -> B T vocab_size

    def forward(self, idx, targets=None):
        
        B, T = idx.shape  # (32, 8)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) = (32, 8, 32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) # broadcasts 
        x = tok_emb + pos_emb # (B,T,C) = (32, 8, 32)
        x = self.blocks(x) # (B,T,C) = (32, 8, 32)
        x = self.ln_f(x) # (B,T,C) = (32, 8, 32)
        logits = self.lm_head(x)  # (B,T,vocab_size) = (32, 8, 64)

        if targets == None: 
            loss = None
        else:
            B, T, C = logits.shape # unpack 
            logits = logits.view(B*T, C) # logits.shape = B*T,C
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens): # ALL IT DOES IS GENERATE THE NEXT TOKEN (CHAR) INDEX 
        # idx is (B,T) array of indices in the current context 
        for _ in range(max_new_tokens): 
            idx_cond = idx[:, -block_size:] # crop context to block size
            logits, loss = self(idx_cond) 
            logits = logits[:, -1, :] # focus on last time step - sets time step to last one in context
            probs = F.softmax(logits, dim=1) 
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1 
        return idx 

# Instantiate class 
model = BigramLanguageModel()
m = model.to(device)



#####  ------------------- 3. OTPIMIZERS + TRAINING LOOP -----------------------

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) 
# Prev. SGD included iteration, forward pass, Linear Layer, BatchNorm, Backward Pass, Update LR, calculate loss w/lr

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets  
    if iter % eval_interval == 0: 
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data 
    xb, yb = get_batch('train') 

    # evaluate the loss 
    logits, loss = m(xb, yb) 
    optimizer.zero_grad(set_to_none=True) 
    loss.backward() 
    optimizer.step() # INTERESTING

# generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device) # starting token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # decode the generated tokens to string