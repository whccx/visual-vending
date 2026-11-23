"""供商品库和识别流程共用的可见像素颜色特征。"""

import base64
import zlib
import numpy as np
import torch

BINS = 16


def descriptor(rgb, weight):
    """计算按可见性权重归一化的 16 维颜色直方图。

    参数：
        rgb: CHW 排列的 RGB 字节张量。
        weight: 像素权重；透明或被掩码排除的像素应为零。
    返回：
        按低饱和度亮度和色相分箱得到的归一化张量。
    """
    rgb = rgb.float()
    mx = rgb.max(0).values
    mn = rgb.min(0).values
    delta = mx - mn
    div = torch.where(delta > 0, delta, torch.ones_like(delta))
    r, g, b = rgb
    hue = torch.zeros_like(mx)
    hue = torch.where((mx == r) & (delta > 0), torch.remainder((g - b) / div, 6), hue)
    hue = torch.where((mx == g) & (delta > 0), (b - r) / div + 2, hue)
    hue = torch.where((mx == b) & (delta > 0), (r - g) / div + 4, hue)
    hue = torch.remainder(hue, 6)
    saturation = torch.where(mx > 0, delta * 255 / mx, torch.zeros_like(mx))
    idx = torch.where(
        saturation < 60,
        torch.clamp((mx / 64).long(), max=3),
        4 + torch.clamp((hue * 2).long(), max=11),
    )
    weights = weight.float()
    q = torch.bincount(idx.flatten(), weights=weights.flatten(), minlength=BINS)
    return q / q.sum().clamp(min=1e-9)


def from_bgra(img):
    """将灰度、BGR 或 BGRA 商品图片转换为颜色特征。

    参数：
        img: OpenCV 格式的图片数组；有 alpha 通道时按透明度加权。
    返回：
        CPU 上的 float32 颜色特征数组。
    异常：
        图片完全透明时抛出 ValueError。
    """
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    alpha = (
        img[:, :, 3].astype(np.float32) / 255
        if img.shape[2] == 4
        else np.ones(img.shape[:2], np.float32)
    )
    if not np.any(alpha):
        raise ValueError("商品图片完全透明")
    rgb = (
        torch.from_numpy(np.ascontiguousarray(img[:, :, :3][:, :, ::-1]))
        .to("cuda")
        .permute(2, 0, 1)
    )
    weight = torch.from_numpy(alpha).to("cuda")
    with torch.inference_mode():
        return descriptor(rgb, weight).cpu().numpy().astype(np.float32)


def encode_mask(mask):
    """将掩码打包、压缩并编码，返回可序列化的尺寸和 Base64 数据。"""
    mask = np.asarray(mask, dtype=np.uint8)
    return {
        "shape": list(mask.shape),
        "bits": base64.b64encode(zlib.compress(np.packbits(mask, axis=None).tobytes())).decode(
            "ascii"
        ),
    }


def decode_mask(encoded):
    """解码 encode_mask 的结果，恢复指定高宽的布尔掩码。"""
    height, width = map(int, encoded["shape"])
    packed = zlib.decompress(base64.b64decode(encoded["bits"]))
    return (
        np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=height * width)
        .reshape(height, width)
        .astype(bool)
    )
