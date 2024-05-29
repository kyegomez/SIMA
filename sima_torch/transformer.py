import torch
from torch import nn, Tensor
from x_transformers import (
    Decoder,
    Encoder,
    TransformerWrapper,
    ViTransformerWrapper,
)


def exists(val):
    return val is not None


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
        image_size: int = 256,
        patch_size: int = 32,
        vit_num_classes: int = 1000,
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
                cross_attend=True,  # set this to True
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

        # Vit
        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=vit_num_classes,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=enc_depth,
                heads=enc_heads,
            ),
        )

    def forward(
        self,
        x,
        img: Tensor = None,
        mask: torch.ones_like = None,
        *args,
        **kwargs,
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

        if exists(img):
            img_embeddings = self.vit(img, return_embeddings=True)
            img_embeddings = self.norm(img_embeddings)
            print(img_embeddings.shape)

            encoded = self.encoder(
                x,
                mask=mask,
                context=img_embeddings,
                return_embeddings=True,
                *args,
                **kwargs,
            )
            print(f"Encoded shape: {encoded.shape}")

            out = self.decoder(
                x, context=encoded, context_mask=mask, *args, **kwargs
            )
            return out
        else:

            encoded = self.encoder(
                x, mask=mask, return_embeddings=True
            )
            out = self.decoder(
                x, context=encoded, context_mask=mask, *args, **kwargs
            )
            return out
