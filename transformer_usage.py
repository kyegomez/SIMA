import torch 
from sima_torch.transformer import SimaTransformer

# Example
x = torch.randint(0, 256, (1, 1024))

# Instantiate the model
model = SimaTransformer(
    dim=512,
    enc_depth=6,
    enc_heads=8,
    dec_depth=6,
    dec_heads=8,
    tie_token_emb=False,
    num_tokens=20000,
    num_memory_tokens=20,
    encoder_dim=512,
    decoder_dim=512,
    max_seq_len=1024,
)

out = model(x)
print(out.shape)  # torch.Size([1, 1024, 512])
