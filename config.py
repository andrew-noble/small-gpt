from dataclasses import dataclass
import torch

DEBUG_MODE = True  # Set this to False for production settings

@dataclass
class GPTConfig:
    # Model configuration
    block_size: int = 64 if DEBUG_MODE else 512
    vocab_size: int = 4096
    n_layer: int = 2 if DEBUG_MODE else 8
    n_head: int = 2 if DEBUG_MODE else 8
    n_embed: int = 64 if DEBUG_MODE else 512
    dropout: float = 0.0 if DEBUG_MODE else 0.2
    bias: bool = False
    use_rotary: bool = False


@dataclass
class TrainingConfig:
    # Training hyperparameters
    learning_rate: float = 6e-4
    max_iters: int = 20 if DEBUG_MODE else 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    decay_lr: bool = False if DEBUG_MODE else True
    warmup_iters: int = 2 if DEBUG_MODE else 1000
    lr_decay_iters: int = 20 if DEBUG_MODE else 30000
    min_lr: float = 6e-5

    # Training loop parameters
    eval_interval: int = 5 if DEBUG_MODE else 100
    log_interval: int = 1 if DEBUG_MODE else 10
    eval_iters: int = 2 if DEBUG_MODE else 200
    gradient_accumulation_steps: int = 1 if DEBUG_MODE else 4
    batch_size: int = 4 if DEBUG_MODE else 64

    # System settings
    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "float32" if DEBUG_MODE else "bfloat16"
    compile: bool = False if DEBUG_MODE else True

# @dataclass
# class GPTConfig:
#     block_size: int = 512
#     vocab_size: int = 4096
#     n_layer: int = 8
#     n_head: int = 8
#     n_embed: int = 512
#     dropout: float = 0.2
#     bias: bool = False
#     use_rotary: bool = False


# @dataclass
# class TrainingConfig:
#     learning_rate: float = 6e-4
#     max_iters: int = 30000
#     weight_decay: float = 1e-1
#     beta1: float = 0.9
#     beta2: float = 0.95
#     grad_clip: float = 1.0

#     decay_lr: bool = True
#     warmup_iters: int = 1000
#     lr_decay_iters: int = 30000
#     min_lr: float = 6e-5

#     eval_interval: int = 100
#     log_interval: int = 10
#     eval_iters: int = 200
#     gradient_accumulation_steps: int = 4
#     batch_size: int = 64

#     device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     dtype: str = "bfloat16"
#     compile: bool = True