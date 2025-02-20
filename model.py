import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from config import GPTConfig

## this is simply the transformer attention layer utilized in the GPT class
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__() #this just reaches up to the parent (nn.Module) class and calls its constructor
        self.config = config
        assert config.n_embed % config.n_head == 0 #this assert syntax... asserts that the number of embeddings is divisible by the number of heads
        self.head_dim = config.n_embed // config.n_head #dimension of each head
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias) #this takes input embedding and expands it into key, value, and query
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias) #this takes expanded atten embedding and projects it back down to the original embedding dimension
        self.attn_dropout = nn.Dropout(config.dropout) #dropout layers randomly kills neurons during training to prevent overfitting
        self.resid_dropout = nn.Dropout(config.dropout)

        #flash attention implementation -- this is an optimization of the attention layer
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print("Not using flash attention")
            self.register_buffer( #this is where we register the bias tensor to set up the biasing behavior of the attention layer. See mask step in attn below
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        # possible future work to add rotary embeddings
        # if config.use_rotary:
        #    self.rotary = Rotary(self.head_dim)

    # SUMMARY of this layer's functionality:
    # 1. pass the tensor through the attention layer, producing queries, keys, and values
    # 2. reshape the queries, keys, and values to prepare for attention
    # 3. apply rotary embeddings if enabled
    # 4. apply flash attention if enabled
    # 5. apply attention pattern to values - this is the math that actually computes the attention
    # 6. reshape the tensor to the original shape of the input tensor, after the attention operations
    # 7. pass the tensor through the residual dropout layer
    # 8. return the output tensor
    def forward(self, x):
        B, T, C = x.shape #shape returns a tuple describing the dims of the tensor. B batch size, T sequence length, C embedding dimension

        q, k, v = self.c_attn(x).split(self.config.n_embed, dim=2) #pass tensor through attention layer and retrieve queries, keys, and values
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) #reshape tensors to prepare for attention
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # Apply rotary embeddings if enabled
        # if self.config.use_rotary:
        #     q, k = self.rotary(q, k)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True,
            )
        else: #this is the attention layer without flash attention
            attn_pattern = (q @ k.transpose(-2, -1)) * ( #just a DOT product of q and k and scaling by the square root of the last dimension of k to prevent exploding gradients
                1.0 / math.sqrt(k.shape[-1])
            )  # B, nh, T, T
            attn_pattern = attn_pattern.masked_fill( #this step masks off parts of the attention pattern that are not allowed to be attented to. Only needed because GPU exposes future tokens, we need to block those
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )
            attn = F.softmax(attn_pattern, dim=-1) #this step applies the softmax function to the attention pattern to get the attention weights
            y = attn @ v  # B, nh, T, T @ B, nh, T, hs -> B, nh, T, hs

        y = y.transpose(1, 2).contiguous().view(B, T, C) #this step reshapes the tensor to the original shape of the input tensor, after the attention operations

        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embed
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.ffd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffd(self.ln_2(x))
        return x