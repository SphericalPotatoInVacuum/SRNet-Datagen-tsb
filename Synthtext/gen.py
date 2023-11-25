# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import random
import time
from pathlib import Path

import Augmentor
import cv2
import numpy as np
from loguru import logger
from pygame import freetype

from . import colorize, data_cfg, render_text_mask


class RetryableError(Exception):
    """Exception raised for errors that are potentially retryable."""

    def __init__(self, message="An error occurred that might be retryable"):
        self.message = message
        super().__init__(self.message)


class Datagen:
    def __init__(self, save_path: Path, seed: int = -1):
        self.save_path = save_path
        if seed == -1:
            seed = time.time_ns() % 2**32
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f'random seed: {seed}')

        freetype.init()

        cur_file_path = Path(__file__).parent

        font_dir = cur_file_path.joinpath(data_cfg.font_dir)
        self.font_list = list(font_dir.iterdir())
        self.standard_font_path = cur_file_path.joinpath(data_cfg.standard_font_path)


        color_filepath = cur_file_path.joinpath(data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)

        text_filepath = cur_file_path.joinpath(data_cfg.text_filepath)
        with open(text_filepath, 'r') as f:
            self.text_list: list[str] = [text.strip() for text in f.readlines()]

        bg_filepath = os.path.join(cur_file_path, data_cfg.bg_filepath)
        with open(bg_filepath, 'r') as f:
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

    def gen_srnet_data_with_background(self, font_path: Path, text: str, idx: int):
        upper_rand = np.random.rand()
        if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
            text = text.capitalize()
        if upper_rand < data_cfg.uppercase_rate:
            text = text.upper()

        # init font
        font = freetype.Font(font_path)
        font.antialiased = True
        font.origin = True

        # choose font style
        font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
        font.underline = np.random.rand() < data_cfg.underline_rate
        font.strong = np.random.rand() < data_cfg.strong_rate
        font.oblique = np.random.rand() < data_cfg.oblique_rate

        # render text to surf
        param = {
            'is_curve': np.random.rand() < data_cfg.is_curve_rate,
            'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() + data_cfg.curve_rate_param[1],
            'curve_center': np.random.randint(0, len(text))
        }
        surf, bbs = render_text_mask.render_text(font, text, param)

        # get padding
        padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
        padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
        padding = np.hstack((padding_ud, padding_lr))

        # perspect the surf
        rotate = data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1]
        zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
        shear = data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1]
        perspect = data_cfg.perspect_param[0] * np.random.randn(2) + data_cfg.perspect_param[1]
        try:
            surf = render_text_mask.perspective(surf, rotate, zoom, shear, perspect, padding) # w first
        except Exception as e:
            raise RetryableError(f"Perspective failed: {e}")

        # choose a background
        surf_h, surf_w = surf.shape[:2]
        surf = render_text_mask.center2size(surf, (surf_h, surf_w))


        while True:
            try:
                bg = cv2.imread(str(random.choice(self.bg_list)))
                bg_h, bg_w = bg.shape[:2]
            except Exception:
                continue
            if surf_w <= bg_w and surf_h <= bg_h:
                break
        x = np.random.randint(0, bg_w - surf_w + 1)
        y = np.random.randint(0, bg_h - surf_h + 1)
        t_b = bg[y:y+surf_h, x:x+surf_w, :]

        # get min h of bbs
        min_h = np.min(bbs[:, 3])

        # get font color
        if np.random.rand() < data_cfg.use_random_color_rate:
            fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)
        else:
            fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)

        # colorize the surf and combine foreground and background
        param = {
            'is_border': np.random.rand() < data_cfg.is_border_rate,
            'bordar_color': tuple(np.random.randint(0, 256, 3)),
            'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
            'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree) + data_cfg.shadow_angle_param[0] * np.random.randn(),
            'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3) + data_cfg.shadow_shift_param[1, :],
            'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn() + data_cfg.shadow_opacity_param[1]
        }
        _, i_s = colorize.colorize(surf, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
        save_path: Path = self.save_path / font_path.stem / f'{text.lower()}_{idx:03}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
