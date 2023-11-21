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
        except RetryableError as e:
            continue
        except Exception as e:
            logger.exception(e)
            raise e


def main():
    datagen = Datagen(Path(cfg.data_dir))
    func = partial(datagen_star, datagen.gen_srnet_data_with_background)
    data = [(font, word, idx)
            for font in datagen.font_list
            for word in random.choices(datagen.text_list, k=cfg.words_per_font)
            for idx in range(cfg.bg_per_word)]
    with mp.Pool(cfg.process_num) as pool:
        list(tqdm(pool.imap_unordered(func, data, chunksize=64), total=len(data)))

if __name__ == '__main__':
    main()
