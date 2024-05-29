import torch
from phenaki_pytorch import CViViT, MaskGit, Phenaki

cvivit = CViViT(
    dim=512,
    codebook_size=65536,
    image_size=(256, 128),  # video with rectangular screen allowed
    patch_size=32,
    temporal_patch_size=2,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=64,
    heads=8,
)

maskgit = MaskGit(
    num_tokens=5000,
    max_seq_len=1024,
    dim=512,
    dim_context=768,
    depth=6,
)

phenaki = Phenaki(cvivit=cvivit, maskgit=maskgit)  #

videos = torch.randn(
    3, 3, 17, 256, 128
)  # # (batch, channels, frames, height, width)
mask = torch.ones(
    (3, 17)
).bool()  # # [optional] (batch, frames) - allows for co-training videos of different lengths as well as video and images in the same batch

texts = [
    "a whale breaching from afar",
    "young girl blowing out candles on her birthday cake",
    "fireworks with blue and green sparkles",
]

loss = phenaki(videos, texts=texts, video_frame_mask=mask)
print(loss.shape)
