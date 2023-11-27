"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import multiprocessing as mp
import random
from functools import partial
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import cfg
from Synthtext.gen import Datagen, RetryableError


def datagen_star(func, args):
    while True:
        try:
            func(*args)
            break
        except RetryableError:
            continue
        except Exception as e:
            logger.exception(e)
            break


def main():
    datagen = Datagen(Path(cfg.data_dir))
    data = [(random.choices(datagen.text_list, k=cfg.images_per_font),) for _ in range(cfg.num_fonts)]
    func = partial(datagen_star, datagen.render_style)
    with mp.Pool(cfg.process_num) as pool:
        list(tqdm(pool.imap_unordered(func, data, chunksize=64), total=len(data)))


if __name__ == '__main__':
    main()
