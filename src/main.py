from utils import UnionConfig, LoggerUtil, DecoratorTimer, PathUtil
from conf import settings
from core.entrypoint import EntryPoint
import os
import argparse
import logging
import shutil
import traceback


def get_args():
    parser = argparse.ArgumentParser()
    str2bool = lambda x: x == 'true'
    parser.add_argument('--dataset_name', type=str, default='amazonVideoGames', help='dataset name')
    parser.add_argument('--task', type=str, default='IR',
                        choices=['IR', 'AI', 'LP', 'Semi-GCN', 'NGCF', 'BPRMF'],
                        help='task_name: [IR, AI, LP, Semi-GCN, NGCF, BPRMF]')
    parser.add_argument('--istrain', type=str2bool, default=True, help='train or test: [true, false]')
    parser.add_argument("--train_id", type=str, default=None, help="item recommendation train task id")
    return parser.parse_args()


def init_all(cfg: UnionConfig):
    cfg.data_folder_path = cfg.DATA_FOLDER_PATH + "/{}".format(cfg.dataset_name)
    cfg.TMPOUT_FOLDER_PATH += "/{}".format(cfg.dataset_name)
    cfg.OUTPUT_FOLDER_PATH += "/{}".format(cfg.dataset_name)
    cfg.TMPOUT_FOLDER_PATH = os.path.realpath(cfg.TMPOUT_FOLDER_PATH)
    cfg.OUTPUT_FOLDER_PATH = os.path.realpath(cfg.OUTPUT_FOLDER_PATH)
    PathUtil.check_path_exist(cfg.data_folder_path)

    if cfg.task == 'IR' and cfg.istrain is True:
        cfg.tmpout_folder_path = cfg.TMPOUT_FOLDER_PATH + "/IR/train/{}".format(cfg.ID)
        cfg.output_folder_path = cfg.OUTPUT_FOLDER_PATH + "/IR/train"
        PathUtil.auto_create_folder_path(
            cfg.tmpout_folder_path + "/pths",
            cfg.output_folder_path
        )
    elif cfg.task == 'IR' and cfg.istrain is False:
        assert cfg.train_id is not None
        cfg.tmpout_folder_path = cfg.TMPOUT_FOLDER_PATH + "/IR/test/{}".format(cfg.ID)
        cfg.output_folder_path = cfg.OUTPUT_FOLDER_PATH + "/IR/test"
        cfg.train_folder_path = cfg.OUTPUT_FOLDER_PATH + "/IR/train/{}".format(cfg.train_id)
        PathUtil.check_path_exist(cfg.train_folder_path + "/all_best_pth.npy")
        PathUtil.auto_create_folder_path(
            cfg.tmpout_folder_path + "/all_metric",
            cfg.output_folder_path
        )
    elif cfg.task in ['AI', 'LP', 'Semi-GCN']:
        cfg.tmpout_folder_path = cfg.TMPOUT_FOLDER_PATH + "/{}/{}".format(cfg.task, cfg.ID)
        cfg.output_folder_path = cfg.OUTPUT_FOLDER_PATH + "/{}".format(cfg.task)
        PathUtil.auto_create_folder_path(
            cfg.tmpout_folder_path,
            cfg.output_folder_path
        )
    elif cfg.task in ['NGCF', 'BPRMF']:
        cfg.tmpout_folder_path = cfg.TMPOUT_FOLDER_PATH + "/{}/{}".format(cfg.task, cfg.ID)
        cfg.output_folder_path = cfg.OUTPUT_FOLDER_PATH + "/{}".format(cfg.task)
        PathUtil.auto_create_folder_path(
            cfg.tmpout_folder_path + "/all_metric",
            cfg.output_folder_path
        )
    else:
        raise Exception("unknown task name")

    log_filepath = cfg.tmpout_folder_path + "/{ID}.log".format(ID=cfg.ID)
    cfg.logger = LoggerUtil(logfile=log_filepath, disableFile=False).get_logger()
    DecoratorTimer.logger = cfg.logger


def main(config):
    config.logger.info("====" * 15)
    config.logger.info("[ID]: " + config.ID)
    config.logger.info("[DATASET]: " + config.dataset_name)
    config.logger.info("[TASK]: " + config.task)
    config.logger.info("[IsTrain]: " + str(config.istrain))
    config.logger.info("[ALL_CFG]: \n" + config.dump_fmt())
    config.dump_file(config.tmpout_folder_path + "/" + "config.json")
    config.logger.info("====" * 15)
    entrypoint = EntryPoint(config)
    entrypoint.start()
    config.logger.info("Task Completed!")


if __name__ == '__main__':
    args = get_args()
    config = UnionConfig.from_py_module(settings)  # get config from settings.py
    config.yml_cfg = UnionConfig.from_yml_file(
        config.CONFIG_FOLDER_PATH + "/datasets/{}.yml".format(args.dataset_name)
    )  # get config from {dataset_name}.yml
    config.merge_asdict(args.__dict__)  # merge config from argparse
    init_all(config)  # init config
    try:
        main(config)
        logging.shutdown()
        shutil.move(config.tmpout_folder_path, config.output_folder_path)
    except:
        config.logger.error(traceback.format_exc())
