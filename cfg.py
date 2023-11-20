"""
Configurations of data generating.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

# dir
data_dir = './output/'
i_t_dir = 'i_t'       # standard text b rendering on gray background
i_s_dir = 'i_s'       # styled text a rendering on background image
t_sk_dir = 't_sk'     # skeletonization of styled text b
t_t_dir = 't_t'       # styled text b rendering on gray background
t_b_dir = 't_b'       # background image
t_f_dir = 't_f'       # styled text b rendering on background image
mask_t_dir = 'mask_t' # the binary mask of styled text b

# sample
sample_num = 100000

# multiprocess
process_num = 16
data_capacity = 256
