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

# a simple layer that applies a sigmoid linear unit activation function and dropout to the input tensor
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
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x))) #sigmoid linear unit activation function

# a composite layer that has the attn layer and silu ff layer and some normalization layers. All this shit is like 100 layer lasagna
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

# the big daddy. This is the overall model, built from the building blocks above
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # register the components of the transformer with pytorch
        transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #weird that its called h but its the list of blocks
            "ln_f": nn.RMSNorm(config.n_embed),
        }

        # Only add positional embeddings if not using rotary
        if not config.use_rotary:
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embed)

        self.transformer = nn.ModuleDict(transformer_dict)

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) #this is output layer which maps the network back down to the vocab size for outputting

        self.transformer.wte.weight = self.lm_head.weight #weight tying -- an optimization

        self.apply(self._init_weights) #apply is inherited from nn.Module

        # this is a special initialization for weights inside the attention layers.
        # they're initialized to a normal distribution with a standard deviation that scales to the number of layers.
        # basically some nerd GPT shit that prevents exploding gradients
        for pn, p in self.named_parameters(): #named_parameters() is inherited from nn.Module
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module): #this initializes weights of the model-- including bias for linear layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)

        # Add learnable positional embeddings
        if not self.config.use_rotary:
            device = idx.device
            b, t = idx.shape
            pos_emb = self.transformer.wpe(
                torch.arange(0, t, dtype=torch.long, device=device)
            )
            x = x + pos_emb

        #this is the actual model, passing the embeddings through each block of layers
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) #normalize the final layer

        #this codeblock just adds in the loss function for training (when there'd be targets), and returns the logits and loss
        if targets is not None:
            logits = self.lm_head(x) #logit is the prediction before being normalized via softmax
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    # side note: a parameter is any trainable value in a network. So weights and biases
    # this method just sets up an optimizer, which is a mechanism for adjusting model weights per the loss function during training
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #we don't decay the bias parameters, which are 1D, hence the dim() check to discriminate decay/no decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        #basically a summary report of the model architecture. snapshot of the model your config specifies
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad() #this decorator tells pytorch to not track gradients for this function
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None
    ):
    #the generation loop
        for _ in range(max_new_tokens):
            context = ( # for this pass, the context is either just this pass (if smaller than block) or everything remaining in the block
                idx
                if idx.size(1) < self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(context)

            logits = logits[:, -1, :] / temperature

            #each of these if branches are sampling tricks to cut down on the number of tokens that are considered for the next token
            #top-p: nucleus sampling. keep smalled set of tokens whos cumulative probability exceeds p
            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs >= top_p
                mask[..., 0] = True

                cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    cut = cutoff_indices[b].item()
                    kept_indices = sorted_indices[b, : cut + 1]
                    top_p_mask[b, kept_indices] = True
                logits[~top_p_mask] = float("-inf")

            #top-k: further restricts the set of tokens to the top k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            #min-p: further restricts the set of tokens to the top p% of tokens
            if min_p is not None and min_p > 0.0:
                logit_max = logits.max(dim=-1, keepdim=True).values
                threshold = logit_max + torch.log(
                    torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
                )
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == 2:
                break
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx