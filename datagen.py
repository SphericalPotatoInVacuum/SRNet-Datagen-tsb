"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import datetime
import logging
import multiprocessing as mp
import random
from functools import partial
from pathlib import Path

import structlog
from tqdm import tqdm

import cfg
from Synthtext.gen import Datagen, RetryableError

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(
    format='%(message)s',
    filename=f'logs/{datetime.datetime.utcnow().isoformat(timespec="seconds")}.jsonl',
    level=logging.DEBUG
)

# Get a logger
logger: structlog.BoundLogger = structlog.get_logger()


def datagen_star(func, args):
    while True:
        try:
            func(*args)
            break
        except RetryableError:
            continue
        except Exception as e:
            logger.error(e)
            break


def main():
    logger.info("Started generating data", num_fonts=cfg.num_fonts, images_per_font=cfg.images_per_font)
    datagen = Datagen(Path(cfg.data_dir))
    data = [(cfg.images_per_font,) for _ in range(cfg.num_fonts)]
    func = partial(datagen_star, datagen.render_style)
    with mp.Pool(cfg.process_num) as pool:
        list(tqdm(pool.imap_unordered(func, data, chunksize=64), total=len(data)))


if __name__ == '__main__':
    main()
