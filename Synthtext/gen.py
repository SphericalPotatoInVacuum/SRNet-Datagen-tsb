# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import uuid4

import Augmentor
import cv2
import numpy as np
from pygame import freetype

from . import colorize, data_cfg, render_text_mask


class RetryableError(Exception):
    """Exception raised for errors that are potentially retryable."""

    def __init__(self, message="An error occurred that might be retryable"):
        self.message = message
        super().__init__(self.message)


@dataclass
class MaskParam:
    is_curve: bool
    curve_rate: float
    curve_center: float


@dataclass
class SurfParam:
    rotate: float
    zoom: np.ndarray
    shear: np.ndarray
    perspect: np.ndarray


@dataclass
class ColorizationParam:
    is_border: bool
    bordar_color: tuple[int, int, int]
    is_shadow: bool
    shadow_angle: float
    shadow_shift: np.ndarray
    shadow_opacity: float


@dataclass
class Style:
    class Capitalization(Enum):
        NONE = 0
        CAPITALIZE = 1
        UPPERCASE = 2

    font: freetype.Font
    capitalization: Capitalization
    padding: np.ndarray
    fg_col: np.ndarray
    bg_col: np.ndarray
    mask_param: MaskParam
    surf_param: SurfParam
    colorization_param: ColorizationParam

    name: str = field(default_factory=lambda: str(uuid4()))


class Datagen:
    def __init__(self, save_path: Path):
        self.save_path = save_path

        freetype.init()

        cur_file_path = Path(__file__).parent

        font_dir = cur_file_path.joinpath(data_cfg.font_dir)
        self.font_list = list(font_dir.iterdir())
        self.standard_font_path = cur_file_path.joinpath(data_cfg.standard_font_path)

        color_filepath = cur_file_path.joinpath(data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)

        text_filepath = cur_file_path.joinpath(data_cfg.text_filepath)
        with open(text_filepath, 'r', encoding='utf-8') as f:
            self.text_list: list[str] = [text.strip() for text in f.readlines()]

        bg_filepath = os.path.join(cur_file_path, data_cfg.bg_filepath)
        with open(bg_filepath, 'r', encoding='utf-8') as f:
            self.bg_list: list[Path] = [Path(img_path.strip()) for img_path in f.readlines()]

        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(
            probability=data_cfg.elastic_rate,
            grid_width=data_cfg.elastic_grid_size,
            grid_height=data_cfg.elastic_grid_size,
            magnitude=data_cfg.elastic_magnitude,
        )

        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(
            probability=data_cfg.brightness_rate,
            min_factor=data_cfg.brightness_min,
            max_factor=data_cfg.brightness_max,
        )
        self.bg_augmentor.random_color(
            probability=data_cfg.color_rate,
            min_factor=data_cfg.color_min,
            max_factor=data_cfg.color_max,
        )
        self.bg_augmentor.random_contrast(
            probability=data_cfg.contrast_rate,
            min_factor=data_cfg.contrast_min,
            max_factor=data_cfg.contrast_max,
        )

    def render_word(self, style: Style, word: str):
        mask_param = style.mask_param.__dict__.copy()
        mask_param['curve_center'] = int(np.round(style.mask_param.curve_center * len(word)))

        surf, bbs = render_text_mask.render_text(style.font, word, mask_param)
        try:
            surf = render_text_mask.perspective(
                surf,
                style.surf_param.rotate,
                style.surf_param.zoom,
                style.surf_param.shear,
                style.surf_param.perspect,
                style.padding,
            )  # w first
        except Exception as e:
            raise RetryableError(f"Perspective failed: {e}") from e

        # choose a background
        surf_h, surf_w = surf.shape[:2]
        surf = render_text_mask.center2size(surf, (surf_h, surf_w))

        t_b = self.gen_bg(surf_w, surf_h)

        # get min h of bbs
        min_h = np.min(bbs[:, 3])

        # colorize the surf and combine foreground and background
        _, i_s = colorize.colorize(surf, t_b, style.fg_col, style.bg_col, self.colorsRGB,
                                   self.colorsLAB, min_h, style.colorization_param.__dict__)

        save_path: Path = self.save_path / style.name / f'{word.lower()}.png'
        cv2.imwrite(str(save_path), i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        return

    def gen_style(self):
        font_path = random.choice(self.font_list)

        # init font
        font = freetype.Font(font_path)
        font.antialiased = True
        font.origin = True

        # choose font style
        font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
        font.underline = np.random.rand() < data_cfg.underline_rate
        font.strong = np.random.rand() < data_cfg.strong_rate
        font.oblique = np.random.rand() < data_cfg.oblique_rate

        # get font color
        fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)

        # get padding
        padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
        padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
        padding = np.hstack((padding_ud, padding_lr))

        # perspect the surf
        surf_param = SurfParam(
            rotate=data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1],
            zoom=data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1],
            shear=data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1],
            perspect=data_cfg.perspect_param[0] * np.random.randn(2) + data_cfg.perspect_param[1],
        )

        # render text to surf
        mask_param = MaskParam(
            is_curve=np.random.rand() < data_cfg.is_curve_rate,
            curve_rate=data_cfg.curve_rate_param[0] * np.random.randn() + data_cfg.curve_rate_param[1],
            curve_center=np.random.rand(),
        )

        capitalization = Style.Capitalization.NONE
        upper_rand = np.random.rand()
        if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
            capitalization = Style.Capitalization.CAPITALIZE
        if upper_rand < data_cfg.uppercase_rate:
            capitalization = Style.Capitalization.UPPERCASE

        colorization_param = ColorizationParam(
            is_border=np.random.rand() < data_cfg.is_border_rate,
            bordar_color=tuple(np.random.randint(0, 256, 3)),
            is_shadow=np.random.rand() < data_cfg.is_shadow_rate,
            shadow_angle=np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree) + data_cfg.shadow_angle_param[0] * np.random.randn(),
            shadow_shift=data_cfg.shadow_shift_param[0, :] * np.random.randn(3) + data_cfg.shadow_shift_param[1, :],
            shadow_opacity=data_cfg.shadow_opacity_param[0] * np.random.randn() + data_cfg.shadow_opacity_param[1],
        )

        return Style(
            font=font,
            capitalization=capitalization,
            padding=padding,
            mask_param=mask_param,
            surf_param=surf_param,
            fg_col=fg_col,
            bg_col=bg_col,
            colorization_param=colorization_param,
        )

    def gen_bg(self, width: int, height: int):
        while True:
            bg = cv2.imread(str(random.choice(self.bg_list)))
            if bg is None:
                continue
            bg_h, bg_w = bg.shape[:2]
            if width <= bg_w and height <= bg_h:
                break

        x = np.random.randint(0, bg_w - width + 1)
        y = np.random.randint(0, bg_h - height + 1)
        t_b = bg[y:y + height, x:x + width, :]

        return t_b

    def render_style(self, words: list[str]):
        style = self.gen_style()
        self.save_path.joinpath(style.name).mkdir(parents=True, exist_ok=True)
        for word in words:
            self.render_word(style, word)
