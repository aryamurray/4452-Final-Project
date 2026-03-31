"""HCCR model implementations.

Provides various architectures for handwritten Chinese character recognition:
- TinyCNNBackbone: Shared CNN backbone with optional CBAM attention
- MiniResNet: Custom residual CNN (~1.56M params, 13 conv layers)
- MiniResNetJoint (mini_resnet): 4-head variant with symbolic scoring
- NeurosymbolicResNet (mini_restnet_join): 7-head neurosymbolic with PhaseManager
- MobileNetV3Wrapper: Transfer learning baseline
- CLIPZeroShot: Zero-shot classification via text-image similarity (requires open_clip)
"""

from .backbone import TinyCNNBackbone
from .mini_resnet import MiniResNet
from .mini_resnet import MiniResNetJoint as MiniResNetJointV1
from .mini_restnet_join import MiniResNetJoint as NeurosymbolicResNet
from .mobilenetv3 import MobileNetV3Wrapper

# CLIPZeroShot requires open_clip which may not be installed
try:
    from .clip_model import CLIPZeroShot
except ImportError:
    CLIPZeroShot = None

__all__ = [
    "TinyCNNBackbone",
    "MiniResNet",
    "MiniResNetJointV1",
    "NeurosymbolicResNet",
    "MobileNetV3Wrapper",
    "CLIPZeroShot",
]
