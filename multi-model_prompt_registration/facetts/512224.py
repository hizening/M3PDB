#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from PIL import Image

def resize_and_convert(input_path: str, output_path: str, size=(224, 224)):
    """
    将 input_path 指定的图片缩放到 size，并以 PNG 格式保存到 output_path。
    """
    with Image.open(input_path) as img:
        # 确保有透明通道（如果不需要透明可改成 "RGB"）
        img = img.convert("RGBA")
        # 用 Resampling 枚举替代旧的 ANTIALIAS
        img_resized = img.resize(size, resample=Image.Resampling.LANCZOS)
        img_resized.save(output_path, format="PNG")
        print(f"[OK] {input_path} → {output_path} (PNG)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将一张 JPG 图片缩放到 224×224 并转换为 PNG")
    parser.add_argument("input", help="原始 JPG 图片路径，如 image.jpg")
    parser.add_argument("output", help="输出 PNG 图片路径，如 image_resized.png")
    args = parser.parse_args()

    resize_and_convert(
        input_path=args.input,
        output_path=args.output,
        size=(224, 224)
    )

