import os
import sys
import time
from functools import wraps
import pytz
import datetime

tensor2npy = lambda x: x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy()
tensor2cpu = lambda x: x.cpu() if x.is_cuda else x


class PathUtil(object):
    @staticmethod
    def auto_create_folder_path(*args):
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path)

    @staticmethod
    def get_main_folder_path():
        return os.path.realpath(os.path.dirname(os.path.abspath(sys.argv[0])))

    @staticmethod
    def check_path_exist(path):
        if not os.path.exists(path):
            raise Exception(os.path.realpath(path) + " not exists")


class IDUtil(object):
    @staticmethod
    def get_random_id_bytime():
        tz = pytz.timezone('Asia/Shanghai')
        return datetime.datetime.now(tz).strftime("%Y%m%d%H%M%S")


class DecoratorTimer:
    logger = None

    def __init__(self, logger=None):
        self.logger = DecoratorTimer.logger or logger

    def msg(self, msg):
        self.logger = DecoratorTimer.logger or self.logger
        if self.logger:
            self.logger.info(msg)
        else:
            print("[INFO]:" + msg)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.msg('函数:{}开始运行...'.format(func.__name__))
            start_time = time.time()
            temp = func(*args, **kwargs)
            end_time = time.time()
            self.msg('函数:{}运行用时：{:.4f}秒'.format(func.__name__, end_time - start_time))
            return temp

        return wrapper
