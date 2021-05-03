import logging
import pytz


class LoggerUtil(object):
    def __init__(self, logfile=None, disableFile=False, name=None):
        self.logfile = logfile
        self.disableFile = disableFile
        self.name = name
        if disableFile is False and logfile is None:
            raise Exception("Please give log filepath")
        self.logger = None
        self._init_logger()

    def _init_logger(self):
        if self.logger is not None:
            return
        logger = logging.getLogger(self.name)

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s(%(name)s)[%(levelname)s]: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        formatter.converter = self.converterCN
        if self.disableFile is False:
            fh = logging.FileHandler(self.logfile, mode='a+', encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger

    @staticmethod
    def converterCN(sec):
        tz = pytz.timezone('Asia/Shanghai')
        dt = pytz.datetime.datetime.fromtimestamp(sec, tz)
        return dt.timetuple()

    def get_logger(self):
        return self.logger
