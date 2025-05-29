import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention, BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention

class ResNetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        num_channels=3,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
        image_size=224,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = [64, 128, 256, 512] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 4, 6, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] = [f"stage{i}" for i in range(1, len(self.depths) + 1)]

class ResNetBlock(nn.module):
    """A single layer of the ResNet architecture."""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            act_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()

        ### Basic conv Layer
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, #kernel_size = 3
            stride=stride, #stride = 1
            padding=kernel_size // 2,
            bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = act_fn()

        self.conv2 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, # kernel_size = 3
            stride=stride, # stride = 1
            padding=kernel_size // 2,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)


        ### ResNet Short Cut
        apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut_layers = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=2, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if not apply_shortcut else self.shortcut_layers

    def forward(self, x):
        # instantiate resiudal
        residual = x
        # convnext layer1
        x = self.conv1(x)
        x = self.normalization(x)
        x = self.activation(x)

        # initial convnext layer2
        x = self.conv1(x)
        x = self.normalization(x)
        
        # Add shortcut
        residual = self.shortcut(residual)
        x = x + residual

        # activation layer
        x = self.activation(x)
        return x
   
class ResNetBottleneck(nn.Module):
    """A single layer of the ResNet architecture."""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            reduction_ratio: int = 4,
            act_fn: nn.Module = nn.ReLU,
            downsample_in_bottleneck: bool = False,
    ):
        super().__init__()
        ### Basic conv Layer
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, #kernel_size = 3
            stride=stride, #stride = 1
            padding=3 // 2,
            bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = act_fn()

        # bottleneck conv layer
        reduce_channels = out_channels // reduction_ratio
        self.conv_bottle = nn.Conv2d(
            in_channels, 
            reduce_channels, 
            kernel_size=1, # kernel_size = 3
            stride=stride, # stride = 1
            padding=3 // 2,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)

        ### ResNet Short Cut
        apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut_layers = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=2, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if not apply_shortcut else self.shortcut_layers

    def forward(self, x):
        # instantiate resiudal
        residual = x

        # convnext bottleneck layer1
        x = self.conv_bottle(x)
        x = self.normalization(x)
        x = self.activation(x)

        # convnext layer2
        x = self.conv1(x)
        x = self.normalization(x)
        x = self.activation(x)

        # convnext layer3
        x = self.conv1(x)
        x = self.normalization(x)

        residual = self.shortcut(residual)
        x = x + residual

        # activation layer
        x = self.activation(x)
        return x
    
class ResNetStage(nn.module):
    """A single stage of the ResNet architecture."""
    def __init__(
            self, 
            config: ResNetConfig,
            in_channels: int,
            out_channels: int,
            stride: int = 2,
            depth: int = 2,
            act_fn: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        layer = ResNetBottleneck if config.layer_type == "bottleneck" else ResNetBlock

        if config.layer_type == "bottleneck":
            first_layer = layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=act_fn(),
                downsample_in_bottleneck=config.downsample_in_bottleneck
            )
        else:
            first_layer = layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=act_fn()
            )
        layers = []
        layers.append(first_layer)
        for _ in range(depth - 1):
            layers.append(
                layer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    activation=act_fn()
                )
            )
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for layer in self.layers:
            x = layer(x)
        return x
    
class ResNetModel(nn.Module):
    """The ResNet model."""
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])

        self.stages.append(
            ResNetStage(
                config=config,
                in_channels=config.embedding_size, # change
                out_channels=config.hidden_sizes[0], # change
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
        ))


        