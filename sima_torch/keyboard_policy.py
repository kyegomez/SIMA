import torch
from torch import nn

from classifier_free_guidance_pytorch import (
    classifier_free_guidance_class_decorator,
)


@classifier_free_guidance_class_decorator
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU()
        )
        self.proj_mid = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU()
        )
        self.proj_out = nn.Linear(dim, 1)

    def forward(
        self,
        inp,
        cond_fns,  # List[Callable] - (1) your forward function now receives a list of conditioning functions, which you invoke on your hidden tensors
    ):
        cond_hidden1, cond_hidden2 = (
            cond_fns  # conditioning functions are given back in the order of the `hidden_dims` set on the text conditioner
        )

        hiddens1 = self.proj_in(inp)
        hiddens1 = cond_hidden1(
            hiddens1
        )  # (2) condition the first hidden layer with FiLM

        hiddens2 = self.proj_mid(hiddens1)
        hiddens2 = cond_hidden2(
            hiddens2
        )  # condition the second hidden layer with FiLM

        return self.proj_out(hiddens2)


# instantiate your model - extra keyword arguments will need to be defined, prepended by `text_condition_`

model = MLP(
    dim=256,
    text_condition_type="film",  # can be film, attention, or null (none)
    text_condition_model_types=(
        "t5",
        "clip",
    ),  # in this example, conditioning on both T5 and OpenCLIP
    text_condition_hidden_dims=(
        512,
        256,
    ),  # and pass in the hidden dimensions you would like to condition on. in this case there are two hidden dimensions (dim * 2 and dim, after the first and second projections)
    text_condition_cond_drop_prob=0.25,  # conditional dropout probability for classifier free guidance. can be set to 0. if you do not need it and just want the text conditioning
)

# now you have your input data as well as corresponding free text as List[str]

data = torch.randn(2, 256)
texts = ["a description", "another description"]

# (3) train your model, passing in your list of strings as 'texts'

# pred = model(data, texts=texts)
# print(pred)
# # after much training, you can now do classifier free guidance by passing in a condition scale of > 1. !

model.eval()
guided_pred = model(
    data, texts=texts, cond_scale=3.0
)  # cond_scale stands for conditioning scale from classifier free guidance paper
print(guided_pred)
