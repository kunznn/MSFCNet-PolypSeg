from .dmcnet_polyp import DMCNetPolyp2d
from .mobilenetv2_unet import MobileNetV2UNet
from .polyp_mfc_net import PolypMFCNet
from .pranet import PraNet
from .pure_unet_2d import PureUNet2D
from .unet import UNet
from .unetpp import UNetPlusPlus2D


def build_model(name: str, in_channels: int = 3, out_channels: int = 1, **kwargs):
    model_name = name.lower()
    if model_name in {"unet", "unet_light", "light_unet"}:
        return UNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"pureunet", "pure_unet", "pureunet2d"}:
        return PureUNet2D(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"unetpp", "unetplusplus", "nested_unet", "nestedunet"}:
        return UNetPlusPlus2D(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"dmcnetfull", "dmcnet_polyp", "dmcnetpolyp2d"}:
        return DMCNetPolyp2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"pranet", "pranet_res2net"}:
        return PraNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"mobilenetv2unet", "mobilenetv2_unet", "mobilenet_unet"}:
        return MobileNetV2UNet(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if model_name in {"polypmfc", "polypmfcnet", "polyp_mfc_net"}:
        return PolypMFCNet(in_channels=in_channels, out_channels=out_channels, **kwargs)

    # TODO: implement attention_unet.
    # TODO: implement litepolypnet.
    available = ["unet", "pureunet", "unetpp", "dmcnetfull", "pranet", "mobilenetv2unet", "polypmfc"]
    raise ValueError(f"Unknown model '{name}'. Available models: {available}")
