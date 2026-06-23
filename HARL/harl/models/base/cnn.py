import torch.nn as nn
from harl.utils.models_tools import init, get_active_func, get_init_method
from harl.models.base.flatten import Flatten

"""CNN Modules."""


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_sizes,
        initialization_method,
        activation_func,
        kernel_size=3,
        stride=1,
        input_scale=255.0,
    ):
        super(CNNLayer, self).__init__()
        self.input_scale = float(input_scale)

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        layers = [
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_sizes[0] // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    hidden_sizes[0]
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    hidden_sizes[0],
                )
            ),
            active_func,
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
            ]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_scale != 1.0:
            x = x / self.input_scale
        x = self.cnn(x)
        return x


class SokobanCNNLayer(nn.Module):
    """CNN encoder for small channel-first symbolic Sokoban boards."""

    def __init__(
        self,
        obs_shape,
        hidden_sizes,
        cnn_channels,
        initialization_method,
        activation_func,
        input_scale=1.0,
    ):
        super().__init__()
        if len(cnn_channels) != 3:
            raise ValueError("Sokoban CNN expects exactly three cnn_channels values.")

        self.input_scale = float(input_scale)
        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(module):
            return init(
                module,
                init_method,
                lambda bias: nn.init.constant_(bias, 0),
                gain=gain,
            )

        self.encoder = nn.Sequential(
            init_(
                nn.Conv2d(
                    obs_shape[0], cnn_channels[0], kernel_size=3, padding=1
                )
            ),
            active_func,
            init_(
                nn.Conv2d(
                    cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1
                )
            ),
            active_func,
            nn.MaxPool2d(kernel_size=2, stride=2),
            init_(
                nn.Conv2d(
                    cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1
                )
            ),
            active_func,
            nn.AdaptiveAvgPool2d((2, 2)),
            Flatten(),
        )

        layers = [
            init_(nn.Linear(cnn_channels[2] * 4, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]
        for index in range(1, len(hidden_sizes)):
            layers.extend(
                [
                    init_(nn.Linear(hidden_sizes[index - 1], hidden_sizes[index])),
                    active_func,
                    nn.LayerNorm(hidden_sizes[index]),
                ]
            )
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_scale != 1.0:
            x = x / self.input_scale
        return self.projection(self.encoder(x))


class CNNBase(nn.Module):
    """A CNN base module for actor and critic."""

    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        architecture = args.get("cnn_architecture", "legacy")
        input_scale = args.get("cnn_input_scale", 255.0)

        if architecture == "sokoban":
            self.cnn = SokobanCNNLayer(
                obs_shape,
                self.hidden_sizes,
                args.get("cnn_channels", [32, 64, 64]),
                self.initialization_method,
                self.activation_func,
                input_scale=input_scale,
            )
        elif architecture == "legacy":
            self.cnn = CNNLayer(
                obs_shape,
                self.hidden_sizes,
                self.initialization_method,
                self.activation_func,
                input_scale=input_scale,
            )
        else:
            raise ValueError(f"Unsupported cnn_architecture: {architecture}")

    def forward(self, x):
        x = self.cnn(x)
        return x
