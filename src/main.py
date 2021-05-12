from utils import UnionConfig, LoggerUtil, DecoratorTimer, PathUtil, add_argument_from_dict_format
from conf import settings
from core.entrypoint import EntryPoint
import os
import argparse
import logging
import shutil
import traceback
import copy
import sys

registered_task_list = ['IR-Train', 'IR-Test', 'AI', 'LP', 'Semi-GCN', 'NGCF', 'BPRMF', 'FM']


def get_config_object_and_parse_args():
    # first time resolve sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='amazonVideoGames', help='dataset name')
    parser.add_argument('--task', type=str, default='IR',
                        choices=registered_task_list,
                        help='task_name: [IR-Train, IR-Test,AI, LP, Semi-GCN, NGCF, BPRMF, FM]')
    parser.add_argument("--train_id", type=str, default=None, help="item recommendation train task id")
    args, unknown_args = parser.parse_known_args()

    config = UnionConfig.from_py_module(settings)  # get config from settings.py
    config.merge_asdict(args.__dict__)  # merge config from argparse
    yml_cfg = UnionConfig.from_yml_file(
        config.CONFIG_FOLDER_PATH + "/datasets/{}.yml".format(args.dataset_name)
    )  # get config from {dataset_name}.yml

    # filter irrelevant config
    tasks = copy.copy(registered_task_list)
    tasks.remove(config.task)
    [yml_cfg.__delitem__(task) for task in tasks if task in yml_cfg.keys()]
    config.yml_cfg = yml_cfg

    # second time resolve sys.argv
    model_cfg = yml_cfg[config.task]
    parser2 = add_argument_from_dict_format(model_cfg, filter_keys=list(args.__dict__.keys()))
    args2 = parser2.parse_args(unknown_args)
    for key in model_cfg.keys():
        if key in args2.__dict__:
            model_cfg[key] = args2.__dict__[key]

    return config


def init_all(cfg: UnionConfig):
    cfg.data_folder_path = cfg.DATA_FOLDER_PATH + "/{}".format(cfg.dataset_name)
    cfg.TMPOUT_FOLDER_PATH += "/{}".format(cfg.dataset_name)
    cfg.OUTPUT_FOLDER_PATH += "/{}".format(cfg.dataset_name)
    cfg.TMPOUT_FOLDER_PATH = os.path.realpath(cfg.TMPOUT_FOLDER_PATH)
    cfg.OUTPUT_FOLDER_PATH = os.path.realpath(cfg.OUTPUT_FOLDER_PATH)
    PathUtil.check_path_exist(cfg.data_folder_path)

    if cfg.task == 'IR-Train':
        cfg.tmpout_folder_path = cfg.TMPOUT_FOLDER_PATH + "/IR/train/{}".format(cfg.ID)
        cfg.output_folder_path = cfg.OUTPUT_FOLDER_PATH + "/IR/train"
        PathUtil.auto_create_folder_path(
            cfg.tmpout_folder_path + "/pths",
            cfg.output_folder_path
        )
    elif cfg.task == 'IR-Test':
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
    elif cfg.task in ['NGCF', 'BPRMF', 'FM']:
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
    config.logger.info("[ARGV]: {}".format(sys.argv))
    config.logger.info("[ALL_CFG]: \n" + config.dump_fmt())
    config.dump_file(config.tmpout_folder_path + "/" + "config.json")
    config.logger.info("====" * 15)
    entrypoint = EntryPoint(config)
    entrypoint.start()
    config.logger.info("Task Completed!")


if __name__ == '__main__':
    config = get_config_object_and_parse_args()
    init_all(config)  # init config
    try:
        main(config)
        logging.shutdown()
        shutil.move(config.tmpout_folder_path, config.output_folder_path)
    except Exception as e:
        config.logger.error(traceback.format_exc())
        raise e
