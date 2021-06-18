import numpy as np
import h5py
import os


class Logger(object):

    def __init__(self, log_fields: (tuple, list) = (), log_foos: (dict, None) = None):
        self.log_fields = None
        self.log_foos = None
        self.log_instructions = None
        self.log_history = None
        self.set_log(log_fields=log_fields, log_foos=log_foos)

    def set_log(self, log_fields, log_foos=()):
        self.log_fields = log_fields
        self.log_history = []

        self.log_foos = log_foos if log_foos is not None else {}
        self.log_instructions = None
        self.set_log_instructions(log_foos)

    def set_log_instructions(self, log_foos):
        self.log_instructions = dict()

        if log_foos not in ({}, (), None):
            for k, v in log_foos.items():
                if not hasattr(log_foos, '__call__'):
                    self.log_instructions[k] = compile(v, "<string>", "eval")
                else:
                    self.log_instructions[k] = v

    def log(self, **kwargs):

        if self.log_fields in ((), None):
            return None

        for k, v in kwargs.items():
            locals()[k] = v

        try:
            log_dict = self.log_history[self.episode]
        except IndexError:
            log_dict = {k: [] for k in self.log_fields}
            self.log_history.append(log_dict)

        values = []
        for k in self.log_fields:
            if k in self.log_instructions:
                foo = self.log_instructions[k]
                v = foo() if hasattr(foo, '__call__') else eval(foo)
            elif k in kwargs:
                v = kwargs[k]
            elif k in self:
                v = getattr(self, k)
            else:
                raise KeyError(f'can\'t find log field `{k}`')

            log_dict[k].append(v)
            values.append(v)

        return values

    def dump_history(self, filename, exist_ok=False):

        if not exist_ok:
            assert not os.path.isfile(filename), f"Specified file `{filename}` exists."

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with h5py.File(filename, 'w') as h5:
            Logger.recursively_save_dict_contents_to_group(h5, '/', Logger.list_to_dict(self.log_history))

    @staticmethod
    def recursively_save_dict_contents_to_group(h5, path, dict_repr):
        """
        ....
        """
        for k, v in dict_repr.items():
            path_k = path + k

            if isinstance(v, np.ndarray) and v.ndim > 2:
                v = [vi for vi in v]

            if isinstance(v, (list, tuple)):
                v = Logger.list_to_dict(v)

            if isinstance(v, (np.ndarray, np.int64, np.float64, str, bytes, int, float)) or v is None:

                if v is None:
                    v = np.nan

                try:
                    h5[path_k] = v
                except (OSError, RuntimeError):
                    if path_k[-1] == ".":
                        path_k = path_k[:-1] + "'.'"
                        h5[path_k] = v

            elif isinstance(v, dict):
                Logger.recursively_save_dict_contents_to_group(h5, path_k + '/', v)
            else:
                raise ValueError(f'Do not understand type {type(v)}.')

    @staticmethod
    def list_to_dict(list_instance):
        return {str(i): vi for i, vi in enumerate(list_instance)}

    @staticmethod
    def dict_to_list(value_dict: dict, key_order=(), recursive=False):
        """ Returns list of value_dict values, ordered by their keys.

            :param value_dict: Dictionary, whose values are returned as list, according to the key_order parameter.
            :param key_order: iterable, defining the order of the values in value_dict in the returned list
                              of values.
                              If no key_order is specified, the keys of the value_dict are converted to a list and
                              sorted.
                              E.g., potential integer keys of a dictionary {0: 'element 0', 1: 'element 1', ...}
                              is converted to a list of ['element 0', 'element 1'], even if the keys are str
                              representations of integer values.
            :param recursive: Whether the dict should recursively be scanned for nested list-like dicts
            """

        if not key_order:

            key_order = []
            for k in value_dict.keys():

                if recursive and isinstance(value_dict[k], dict):
                    v = Logger.dict_to_list(value_dict[k], recursive=True)
                    value_dict[k] = v

                try:
                    key_order.append(int(k))
                except (TypeError, ValueError):
                    key_order.append(k)

            key_order = np.asarray(key_order)
            try:
                assert not any(isinstance(k, str) for k in key_order)
                return [list(value_dict.values())[k] for k in key_order.argsort()]
            except AssertionError:
                return value_dict

        return [value_dict[k] for k in key_order]

    def load_history(self, filename) -> list:
        with h5py.File(filename, 'r') as h5:
            loaded = Logger.recursively_load_dict_contents_from_group(h5, '/')
            loaded = Logger.dict_to_list(loaded, recursive=True)
            self.log_history = [Logger.wrap_np(li) for li in loaded]
            return self.log_history

    @staticmethod
    def recursively_load_dict_contents_from_group(h5, path='/') -> dict:
        """
        ....
        """
        ans = {}
        for key, item in h5[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]  # access item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = Logger.recursively_load_dict_contents_from_group(h5, path + key + '/')

        return ans

    @staticmethod
    def wrap_np(log_dict):
        for k in log_dict.keys():
            log_dict[k] = np.asarray(log_dict[k])

        return log_dict
