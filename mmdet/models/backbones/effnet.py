from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from ..utils import build_conv_layer, build_norm_layer
from ..registry import BACKBONES


class MBConvBlockList(nn.Module):
    def __init__(self):
        self.blocks = []
        pass

    def add_block(self, block):
        self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, x):
        return self.forward(x)


@BACKBONES.register_module
class EffNet(nn.Module):
    def __init__(self,
                 eff_name,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4, 5, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):
        super(EffNet, self).__init__()
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        model = EfficientNet.from_pretrained(eff_name)
        self.block_lists, self.layer_name_list = self._make_effnet_layers(model)
        out_channels = self.block_lists[0].blocks[0]._block_args.input_filters
        self._make_stem_layer(in_channels, out_channels)

    def init_weights(self, pretrained=None):
        pass

    def _make_stem_layer(self, in_channels, out_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, self.norm1 = build_norm_layer(self.norm_cfg, out_channels, postfix=1)
        self.add_module(self.norm1_name, self.norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_effnet_layers(self, model):
        block_lists = []
        layer_name_list = []
        block_idxs = {}
        for i, block in enumerate(model._blocks):
            bargs = block._block_args
            outf = bargs.output_filters
            if outf not in block_idxs:
                block_idxs[outf] = [i]
            else:
                block_idxs[outf].append(i)
        blocks = model._blocks
        for i, (k, v) in enumerate(block_idxs.items()):
            bl = MBConvBlockList()
            for idx in v:
                bl.add_block(blocks[idx])
            layer = nn.Sequential(*bl.blocks)
            layer_name = 'layer{}'.format(i)
            self.add_module(layer_name, layer)
            block_lists.append(bl)
            layer_name_list.append(layer_name)
        return block_lists, layer_name_list

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.layer_name_list):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(EffNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
