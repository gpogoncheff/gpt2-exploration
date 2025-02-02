from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # Q, K, V projections for all heads
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        # Output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        #mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch, seq len, embed dim
        # query, key, and values for all heads in batch and move head to be in the batch
        # nh: number of heads, hs: head size, C: num_channels = nh*ns
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (T, T) matrix for all queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq len
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained gpt2 model weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embed": 768},          # 124M Params
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embed": 1024},  # 350M Params
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embed": 1280},   # 774M Params
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embed": 1600},      # 1558M Params
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith(".attn.bias")]

        hugging_face_model = GPT2LMHeadModel.from_pretrained(model_type)
        hugging_face_state_dict = hugging_face_model.state_dict()

        # Load weights from huggingface model and ensure that the weights are compatible
        hugging_face_state_dict_keys = hugging_face_state_dict.keys()
        hugging_face_state_dict_keys = [k for k in hugging_face_state_dict_keys if not k.endswith(".attn.masked_bias")]
        hugging_face_state_dict_keys = [k for k in hugging_face_state_dict_keys if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(state_dict_keys) == len(hugging_face_state_dict_keys)
        for k in hugging_face_state_dict_keys:
            if any(k.endswith(w) for w in transposed):
                assert hugging_face_state_dict[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(hugging_face_state_dict[k].t())
            else:
                assert hugging_face_state_dict[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(hugging_face_state_dict[k])

        return model

    def forward(self, idx):
        # idx: token indices of shape (B, T), batch size by seq len
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence length exceeds block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # position
        pos_embed = self.transformer.wpe(pos) # shape (T, n_embed)
        token_embed = self.transformer.wte(idx) # shape (B, T, n_embed)
        x = token_embed + pos_embed
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # final layer norm
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        return logits




if __name__ == "__main__":
    import tiktoken

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.cuda.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    num_return_sequences = 5
    max_length = 30

    model = GPT(GPTConfig()) # randomly initialized model
    #model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, my name is gpt2. I am a language model. If you are seeing this message,")
    tokens = torch.tensor(tokens, dtype=torch.long) # shape (num_input_tokens, )
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # shape (num_return_sequences, num_input_tokens)
    x = tokens.to(device)

    torch.manual_seed(13)
    torch.cuda.manual_seed(13)
    while x.size(1) < max_length:
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # get last predicted token (B, vocab_size)
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        # top-k sampling of 50 (in accordance with hugging face)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
        # select token from topk_probs
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print("> ", decoded)
