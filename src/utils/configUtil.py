# -*- coding: utf-8 -*-
"""
@Author: kervias
"""

import copy
import configparser
import os
import json
import yaml


class UnionConfig(object):
    """
        统一配置文件对象
    """

    def __init__(self, dic: dict = dict()):
        self.__config__ = dic

    @classmethod
    def from_ini_file(cls, inifilepath, convert=True, strip=True):
        config = configparser.ConfigParser()
        if not os.path.exists(inifilepath):
            raise Exception("%s not found" % inifilepath)
        config.read(inifilepath, encoding='utf-8')
        dic = {k: dict(config.items(k)) for k in config.sections()}
        if convert is True:
            for k1, v1 in dic.items():
                for k2, v2 in v1.items():
                    ind = v2.find(' ')
                    assert ind != -1
                    type_ = v2[0:ind + 1].strip() if strip else v2[0:ind + 1]
                    v2 = v2[ind + 1::].strip() if strip else v2[ind + 1::]
                    if type_ in ['bool', 'int', 'float']:
                        v1[k2] = eval("%s(%s)" % (type_, v2))
                    elif type_ == 'str':
                        v1[k2] = eval("%s('%s')" % (type_, v2))
                    elif type_ == 'json':
                        v1[k2] = json.loads(v2)
                    else:
                        raise Exception("convert failed: unknown type <%s>" % type_)
                dic[k1] = v1
        return cls(dic)

    @classmethod
    def from_py_module(cls, module_object):
        assert str(type(module_object)) == "<class 'module'>"
        return cls(
            {k: (getattr(module_object, k) if not k.endswith("_PATH") else os.path.realpath(getattr(module_object, k)))
             for k in dir(module_object) if k[0].isupper()}
        )

    @classmethod
    def from_yml_file(cls, filepath):
        config = dict()
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def __iter__(self):
        for k in self.__config__.keys():
            yield k

    def __getattr__(self, key):
        if key in self.__config__.keys():
            return self.__config__[key]
        elif key in dir(self):
            return self.__dict__[key]
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, key))

    def __setattr__(self, key, value):

        if key != "__config__":
            self.__config__[key] = value
        else:
            self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__config__.keys():
            del self.__config__[key]
        elif key in dir(self):
            raise Exception("attribute '%s' is not allowed to delete" % key)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__base__, key))

    def __setitem__(self, key, value):
        assert key not in dir(self), "conflict with dir(self)"
        self.__config__[key] = value

    def __getitem__(self, key):
        return self.__config__[key]

    def __delitem__(self, key):
        del self.__config__[key]

    def keys(self):
        return self.__config__.keys()

    def items(self):
        return self.__config__.items()

    # def as_dict(self):
    #     return copy.deepcopy(self.__config__)

    def get(self, key: str, default_value=None):
        return self.__config__.get(key, default_value)

    def merge_asdict(self, obj: object):
        intersec = set(self.keys()).intersection(obj.keys())
        if len(intersec) > 0:
            raise Exception("Merge failed: two object keys has same key: %s" % intersec)
        for k, v in obj.items():
            self[k] = v

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.__config__))

    def __repr__(self):
        return self.__str__()

    def dump_fmt(self):
        default_func = lambda o: o.__config__ if isinstance(o, UnionConfig) else str(o)
        return json.dumps(self.__config__, indent=4, ensure_ascii=False, default=default_func)

    def dump_file(self, filepath):
        default_func = lambda o: o.__config__ if isinstance(o, UnionConfig) else str(o)
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(self.__config__, f, indent=4, ensure_ascii=False, default=default_func)