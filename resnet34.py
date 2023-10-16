# %%
import json
import os
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Optional, Union
import requests
import torch as t
import torchvision
from einops import rearrange
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn.functional import conv1d as torch_conv1d
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

import utils

MAIN = __name__ == "__main__"

# %%
URLS = [
    "https://www.oregonzoo.org/sites/default/files/styles/article-full/public/animals/H_chimpanzee%20Jackson.jpg",
    "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg",
    "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg",
    "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg",
    "http://www.tudobembresil.com/wp-content/uploads/2015/11/nouvelancopacabana.jpg",
    "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg",
    "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg",
    "https://i.redd.it/mbc00vg3kdr61.jpg",
    "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg",
]


def load_image(url: str) -> Image.Image:
    """Return the image at the specified URL, using a local cache if possible.

    Note that a robust implementation would need to take more care cleaning the image names.
    """
    os.makedirs("./images", exist_ok=True)
    filename = os.path.join("./images", url.rsplit("/", 1)[1].replace("%20", ""))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = f.read()
    else:
        response = requests.get(url)
        data = response.content
        with open(filename, "wb") as f:
            f.write(data)
    return Image.open(BytesIO(data))


images = [load_image(url) for url in tqdm(URLS)]
display(images[0])

# %%
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# %%
def prepare_data(images: list[Image.Image]) -> t.Tensor:
    """Preprocess each image and stack them into a single tensor.

    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    """
    x = t.stack([preprocess(img) for img in tqdm(images)], dim=0)  # type: ignore
    return x


prepare_data(images)
# %%
with open("imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())


def predict(model, images: list[Image.Image], print_topk_preds=3) -> list[float]:
    """
    Pass the images through the model and print out the top predictions.

    For each image, `display()` the image and the most likely categories according to the model.

    Return: for each image, the index of the top prediction.
    """
    model.eval()
    images_norm = prepare_data(images)
    with t.inference_mode():
        preds = model(images_norm)

    preds = preds.softmax(dim=1)
    _, indices = t.topk(preds, print_topk_preds, dim=1)
    for i, image in enumerate(images):
        small = image.copy()
        small.thumbnail((150, 150))
        display(small)
        print(
            "\n".join(
                f"{100*preds[i][j]:.4f}% {imagenet_labels[j]}" for j in indices[i]
            )
        )

    return [idx[0].item() for idx in indices]


if MAIN:
    model = models.resnet34(weights="DEFAULT")
    pretrained_categories = predict(model, images)
    print(pretrained_categories)


# %%
def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    assert x.shape[1] == weights.shape[1]
    B, Cin, W = x.shape
    _, _, KW = weights.shape

    BS, CinS, WS = x.stride()  # type: ignore

    x_strided = x.as_strided(
        size=(B, Cin, W - KW + 1, KW), stride=(BS, CinS, WS, WS)
    )  # The third stride adjusts *actual* stride!
    conv_w = t.einsum("bijk,oik->boj", x_strided, weights)
    return conv_w


# %%
def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    """Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = weights.shape

    BS, CinS, HS, WS = x.stride()  # type: ignore

    x_strided = x.as_strided(
        size=(B, Cin, H - KH + 1, W - KW + 1, KH, KW), stride=(BS, CinS, HS, WS, HS, WS)
    )
    conv_w = t.einsum("bchwij,ocij->bohw", x_strided, weights)
    return conv_w


# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    y = t.zeros((x.shape[0], x.shape[1], left + x.shape[2] + right))
    print(y.shape, x.shape)
    y[:, :, : left + 1] = pad_value
    y[:, :, left + x.shape[2] :] = pad_value
    y[:, :, left : left + x.shape[2]] = x
    return y


# %%
def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    B, C, H, W = x.shape
    out = x.new_full((B, C, top + H + bottom, left + W + right), pad_value)
    out[..., top : top + H, left : left + W] = x
    # y = (
    #     t.ones((x.shape[0], x.shape[1], top + x.shape[2] + bottom, left + x.shape[3] + right), dtype=t.float32)
    #     * pad_value
    # )
    # y[:, :, top : x.shape[2] + top, left : x.shape[3] + left] = x
    return out


# %%
def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    """Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    B, Cin, W = x.shape
    _, _, KW = weights.shape
    OW = (W + 2 * padding - KW) // stride + 1

    x = pad1d(x, padding, padding, 0.0)
    BS, CinS, WS = x.stride()  # type: ignore

    x_strided = x.as_strided(
        size=(B, Cin, OW, KW), stride=(BS, CinS, stride, WS)
    )  # The third stride adjusts *actual* stride!
    conv_w = t.einsum("bijk,oik->boj", x_strided, weights)
    return conv_w


# %%
IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0):
    """Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    """
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = weights.shape
    OH = (H + 2 * force_pair(padding)[0] - KH) // force_pair(stride)[0] + 1
    OW = (W + 2 * force_pair(padding)[1] - KW) // force_pair(stride)[1] + 1

    x = pad2d(
        x,
        force_pair(padding)[1],
        force_pair(padding)[1],
        force_pair(padding)[0],
        force_pair(padding)[0],
        0.0,
    )

    BS, CinS, HS, WS = x.stride()  # type: ignore

    x_strided = x.as_strided(
        size=(B, Cin, OH, OW, KH, KW),
        stride=(BS, CinS, HS * force_pair(stride)[0], force_pair(stride)[1], HS, WS),
    )
    conv_w = t.einsum("bchwij,ocij->bohw", x_strided, weights)
    return conv_w


# %%
def maxpool2d(
    x: t.Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> t.Tensor:
    """Like torch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, height, width)
    """
    B, Cin, H, W = x.shape
    kH, kW = force_pair(kernel_size)
    sH, sW = force_pair(stride) if stride != None else (kH, kW)
    pH, pW = force_pair(padding)[0], force_pair(padding)[1]
    OH = ((H + 2 * pH - kH) // sH) + 1
    OW = ((W + 2 * pW - kW) // sW) + 1

    x = pad2d(x, pW, pW, pH, pH, float("-inf"))

    BS, CinS, HS, WS = x.stride()  # type: ignore

    x_strided = x.as_strided(
        size=(B, Cin, OH, OW, kH, kW), stride=(BS, CinS, HS * sH, WS * sW, HS, WS)
    )
    return x_strided.amax(dim=(-2, -1))


# %%
def extra_repr(obj, args: list[str], kwargs: list[str]) -> str:
    reprs = [repr(getattr(obj, arg)) for arg in args] + [
        f"{karg}={getattr(obj, karg)}" for karg in kwargs
    ]
    return ", ".join(reprs)


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return extra_repr(self, [], ["kernel_size", "stride", "padding"])


# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        bound = (2 / in_features) ** 0.5  # Kaiming initalization!?
        self.weight = nn.Parameter(
            t.empty((out_features, in_features)).normal_(mean=0, std=bound)
        )
        self.bias = (
            nn.Parameter(t.zeros(out_features).normal_(mean=0, std=bound))
            if bias
            else None
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = t.einsum("...j,kj->...k", x, self.weight)
        if self.bias != None:
            return x + self.bias
        return x

    def extra_repr(self) -> str:
        if self.bias is not None:
            return extra_repr(self, [], ["weight", "bias"])
        else:
            return extra_repr(self, [], ["weight"])


# %%
class Conv2d(t.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        bound = (2 / in_channels) ** 0.5
        kH, kW = force_pair(kernel_size)
        self.weight = nn.Parameter(
            t.empty((out_channels, in_channels, kH, kW)).normal_(mean=0, std=bound)
        )
        self.stride = stride
        self.padding = padding
        self.bias = False
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        """"""
        return extra_repr(self, [], ["weight", "stride", "bias"])


# %%
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor
    "running_mean: shape (num_features,)"
    running_var: t.Tensor
    "running_var: shape (num_features,)"
    num_batches_tracked: t.Tensor
    "num_batches_tracked: shape ()"

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            mean = x.mean((0, 2, 3))
            var = x.var((0, 2, 3), unbiased=False)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1

        else:
            mean = self.running_mean
            var = self.running_var

        rs = lambda t: t.reshape(1, -1, 1, 1)  # I hate this so much dim ruined my day
        return ((x - rs(mean)) / t.sqrt(rs(var) + self.eps)) * rs(self.weight) + rs(
            self.bias
        )

    def extra_repr(self) -> str:
        return extra_repr(self, ["num_features"], ["momentum", "eps"])


# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


# %%
class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(f"{m._get_name()}_{i}", m)

    def forward(self, x: t.Tensor):
        """Chain each module together, with the output from one feeding into the next one"""
        for v in self._modules.values():
            assert (
                v != None
            ), "self._modules can contain None in general, but Sequential shouldn't!"
            x = v(x)
        return x


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """Flatten out dimensions from start_dim to end_dim, inclusive of both.

        Return a view if possible, otherwise a copy.
        """
        end_dim = self.end_dim + (len(input.shape) if self.end_dim < 0 else 0) + 1
        output_shape = (
            [input.shape[i] for i in range(0, self.start_dim)]
            + [-1]
            + [input.shape[i] for i in range(end_dim, len(input.shape))]
        )
        return input.reshape(output_shape)

    def extra_repr(self) -> str:
        return extra_repr(self, [], ["start_dim", "end_dim"])


# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """
        return x.mean((-2, -1))


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        self.left = Sequential(
            Conv2d(
                in_feats,
                out_feats,
                kernel_size=3,
                stride=first_stride,
                padding=1,
            ),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(
                out_feats,
                out_feats,
                kernel_size=3,
                padding=1,
            ),
            BatchNorm2d(out_feats),
        )

        self.right = (
            Sequential(
                Conv2d(
                    in_feats,
                    out_feats,
                    kernel_size=1,
                    stride=first_stride,
                ),
                BatchNorm2d(out_feats),
            )
            if first_stride != 1
            else None
        )

        self.out_relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        """
        l_x = self.left(x)
        r_x = x if self.right is None else self.right(x)
        out = self.out_relu(l_x + r_x)
        return out


# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()
        self.blocks = Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *(ResidualBlock(out_feats, out_feats) for _ in range(n_blocks - 1)),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Compute the forward pass.
        x: shape (batch, in_feats, height, width)
        Return: shape (batch, out_feats, height/stride, width/stride)

        If no downsampling block is present, the addition should just add the left branch's output
        to the input.
        """
        return self.blocks(x)


# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_features_per_group = [64] + out_features_per_group[:-1]
        self.model = Sequential(
            Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, stride=2, padding=1),
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    in_features_per_group,
                    out_features_per_group,
                    strides_per_group,
                )
            ),
            AveragePool(),
            Flatten(),
            Linear(512, n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape(batch, n_classese)
        """
        return self.model(x)


# %%
if MAIN:
    my_model = ResNet34()
    pre_model = models.resnet34(weights="DEFAULT")
    pre_model_state = pre_model.state_dict().items()
    my_model_state = my_model.state_dict().items()
    assert len(pre_model_state) == len(
        my_model_state
    ), "Differing number of saved state tensors!"

    import pandas as pd  # useful trick!

    df = pd.DataFrame.from_records(
        [
            (tk, tuple(tv.shape), mk, tuple(mv.shape))
            for ((tk, tv), (mk, mv)) in zip(pre_model_state, my_model_state)
        ],
        columns=[
            "pre model name",
            "pre model shape",
            "my model name",
            "my model shape",
        ],
    )
    with pd.option_context("display.max_rows", None):
        display(df)

    assert all(
        x[1].shape == y[1].shape for x, y in zip(pre_model_state, my_model_state)
    ), "Shapes don't match up!"
    d = OrderedDict(
        (my_state[0], pre_state[1])
        for my_state, pre_state in zip(my_model_state, pre_model_state)
    )
    my_model.load_state_dict(d, strict=True)


# %%
def check_nan_hook(module: nn.Module, inputs, output):
    """Example of a hook function that can be registered to a module."""
    x = inputs[0]
    if t.isnan(x).any():
        raise ValueError(module, x)
    out = output[0]
    if t.isnan(out).any():
        raise ValueError(module, x)


def add_hook(module: nn.Module) -> None:
    """Remove any exisisting hooks and register our hook.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    """

    utils.remove_hooks(module)
    module.register_forward_hook(check_nan_hook)


if MAIN:
    my_model.apply(add_hook)
    my_model_pred = predict(my_model, images)
# %%
cifar_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def get_cifar10():
    """Download (if necessary) and return the CIFAR10 dataset."""

    mean = t.tensor([125.307, 122.961, 113.8575]) / 255
    std = t.tensor([51.5865, 50.847, 51.255]) / 255
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        "cifar10_train", transform=transform, download=True, train=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        "cifar10_test", transform=transform, download=True, train=False
    )
    return cifar_train, cifar_test


if MAIN:
    cifar_train, cifar_test = get_cifar10()
    trainloader = DataLoader(cifar_train, batch_size=512, shuffle=True, pin_memory=True)
    testloader = DataLoader(cifar_test, batch_size=512, pin_memory=True)
# %%
if MAIN:
    batch = next(iter(trainloader))
    print("Mean value of each channel: ", batch[0].mean((0, 2, 3)))
    print("Std value of each channel: ", batch[0].std((0, 2, 3)))
    (fig, axs) = plt.subplots(ncols=5, figsize=(15, 5))
    for i, ax in enumerate(axs):
        ax.imshow(rearrange(batch[0][i], "c h w -> h w c"))
        ax.set(xlabel=cifar_classes[batch[1][i].item()])
# %%
MODEL_FILENAME = "./resnet34_cifar10.pt"
device = "cuda" if t.cuda.is_available() else "cpu"


def train(trainloader: DataLoader, epochs: int) -> ResNet34:
    model = ResNet34(n_classes=10).to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_fn = t.nn.CrossEntropyLoss()
    for e in range(epochs):
        for i, (x, y) in enumerate(tqdm(trainloader)):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {e}, train loss is {loss}")
        print(f"Saving model to: {os.path.abspath(MODEL_FILENAME)}")
        t.save(model, MODEL_FILENAME)
    return model


# %%
if MAIN:
    if os.path.exists(MODEL_FILENAME):
        print("Loading model from disk: ", MODEL_FILENAME)
        model = t.load(MODEL_FILENAME)
    else:
        print("Training model from scratch")
        model = train(trainloader, epochs=8)
# %%
if MAIN:
    model.eval()
    model.apply(add_hook)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")
    with t.inference_mode():
        n_correct = 0
        n_total = 0
        loss_total = 0.0
        for i, (x, y) in enumerate(tqdm(testloader)):
            x = x.to(device)
            y = y.to(device)
            with t.autocast(device):
                y_hat = model(x)
                loss_total += loss_fn(y_hat, y).item()
            n_correct += (y_hat.argmax(dim=-1) == y).sum().item()
            n_total += len(x)
        print(
            f"Test accuracy: {n_correct} / {n_total} = {100 * n_correct / n_total:.2f}%"
        )
        print(f"Test loss: {loss_total / n_total}")
