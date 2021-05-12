import argparse


def add_argument_from_dict_format(cfg: dict, filter_keys=[]):
    parser = argparse.ArgumentParser()
    str2bool = lambda x: x == 'true'

    def str2list_type(type_=str):
        return lambda string: [type_(item.strip()) for item in string.split(',')]

    for k, v in cfg.items():
        if type(v) not in [str, list, float, bool, int] or type(k) != str or k in filter_keys:
            print("ignore resolve {}".format(k))
            continue
        if type(v) == bool:
            parser.add_argument('--' + k, type=str2bool, default=v)
        elif type(v) == list:
            assert len(v) > 0 and type(v[0]) in [int, float, str]
            parser.add_argument('--' + k, type=str2list_type(type(v[0])), default=v)
        else:
            parser.add_argument('--' + k, type=type(v), default=v)
    return parser
