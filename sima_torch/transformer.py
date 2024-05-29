import torch
from torch import nn
from x_transformers import (
    Decoder,
    Encoder,
    TransformerWrapper,
)


class SimaTransformer(nn.Module):
    """
    SimaTransformer is a transformer-based model for sequence-to-sequence tasks.

    Args:
        dim (int): The dimensionality of the model. Default is 512.
        enc_depth (int): The depth of the encoder. Default is 6.
        enc_heads (int): The number of attention heads in the encoder. Default is 8.
        dec_depth (int): The depth of the decoder. Default is 6.
        dec_heads (int): The number of attention heads in the decoder. Default is 8.
        tie_token_emb (bool): Whether to tie the token embeddings of the encoder and decoder. Default is False.
        num_tokens (int): The number of tokens in the vocabulary. Default is 20000.
        num_memory_tokens (int): The number of memory tokens. Default is 20.
        encoder_dim (int): The dimensionality of the encoder. Default is 512.
        decoder_dim (int): The dimensionality of the decoder. Default is 512.
        max_seq_len (int): The maximum sequence length. Default is 1024.
        training_on (bool): Whether the model is in training mode. Default is False.

    Attributes:
        encoder (TransformerWrapper): The encoder module of the transformer.
        decoder (TransformerWrapper): The decoder module of the transformer.
        norm (nn.LayerNorm): The layer normalization module.

    """

    def __init__(
        self,
        dim: int = 512,
        enc_depth: int = 6,
        enc_heads: int = 8,
        dec_depth: int = 6,
        dec_heads: int = 8,
        tie_token_emb: bool = False,
        num_tokens: int = 20000,
        num_memory_tokens: int = 20,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        max_seq_len: int = 1024,
        training_on: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.enc_depth = enc_depth
        self.enc_heads = enc_heads
        self.dec_depth = dec_depth
        self.dec_heads = dec_heads
        self.tie_token_emb = tie_token_emb
        self.num_tokens = num_tokens
        self.num_memory_tokens = num_memory_tokens
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_seq_len = max_seq_len
        self.training_on = training_on

        self.encoder = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            num_memory_tokens=num_memory_tokens,  # 20 memory tokens
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=enc_depth,
                heads=enc_heads,
                *args,
                **kwargs,
            ),
        )

        # Decoder
        self.decoder = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=False,  # set this to False
            post_emb_norm=True,  # set this to True to layernorm summed token + pos embeddings
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=dec_depth,
                heads=dec_heads,
                # attn_qk_norm = True,
                # attn_qk_norm_dim_scale = True, # set this to True, in addition to `attn_qk_norm = True`
                ff_swish=True,  # set this to True
                ff_glu=True,  # set to true to use for all feedforwards
                # attn_kv_heads = 4,
                cross_attend=True,  # set this to True
                *args,
                **kwargs,
            ),
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, x, mask: torch.ones_like = None, *args, **kwargs
    ):
        """
        Forward pass of the SimaTransformer model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len). Default is None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, dim).

        """
        if mask is None:
            mask = torch.ones_like(x).bool()

        if self.training_on:
            # Get the loss
            pass
        else:
            encoded = self.encoder(
                x, mask=mask, return_embeddings=True
            )
            out = self.decoder(
                x, context=encoded, context_mask=mask, *args, **kwargs
            )
            return out


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
