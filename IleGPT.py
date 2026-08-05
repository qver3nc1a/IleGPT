import torch
import tiktoken
import torch.nn as nn
from TransformerBlock import TransformerBlock
from Chapter4 import LayerNorm, GPT_CONFIG_124M


class IleGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

torch.manual_seed(123)
model = IleGPT(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# weight tying
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(
    f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
)

# parameters in feedforward and attention

block = TransformerBlock(GPT_CONFIG_124M)
total_params_ff = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params_ff:,}")

total_params_att = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params_att:,}")

# GPT-2 large
# GPT_CONFIG["emb_dim"] = 1600
# GPT_CONFIG["n_layers"] = 48
# GPT_CONFIG["n_heads"] = 25
# model = GPTModel(GPT_CONFIG)


# generating text
def generate_text_simple(
    model, idx, max_new_tokens, context_size
):  # idx is a (batch, n_tokens) array of indices

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # crop content if exceeds context size
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)  # shape (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # shape (batch, 1)
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # appends sampled idx to running sequence with idx shape (batch, n_tokens+1)
    return idx


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()  # disables random components (here dropout)
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
