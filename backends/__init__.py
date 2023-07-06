import abc
import importlib
import inspect
import json
import sys
import os
import logging
import logging.config
from typing import Dict, Callable, List, Tuple, Any

import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
with open(os.path.join(project_root, "logging.yaml")) as f:
    conf = yaml.safe_load(f)
    log_fn = conf["handlers"]["file_handler"]["filename"]
    log_fn = os.path.join(project_root, log_fn)
    conf["handlers"]["file_handler"]["filename"] = log_fn
    logging.config.dictConfig(conf)


def get_logger(name):
    return logging.getLogger(name)


# Load backend dynamically from "backends" sibling directory
# Note: The backends might use get_logger (circular import)
def load_credentials(backend, file_name="key.json") -> Dict:
    key_file = os.path.join(project_root, file_name)
    with open(key_file) as f:
        creds = json.load(f)
    assert backend in creds, f"No '{backend}' in {file_name}. See README."
    assert "api_key" in creds[backend], f"No 'api_key' in {file_name}. See README."
    return creds


class Backend(abc.ABC):

    @abc.abstractmethod
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[Any, Any, str]:
        pass

    @abc.abstractmethod
    def supports(self, model_name: str):
        pass

    def __repr__(self):
        return f"Backend({str(self)})"

    def __str__(self):
        return self.__class__.__name__.lower()


def is_backend(obj):
    if inspect.isclass(obj) and issubclass(obj, Backend):
        return True
    return False


# We load all _api.py-modules in the backends directory and load all classes that implement Backend
_loaded_backends: List[Backend] = []
backends_root = os.path.join(project_root, "backends")
if os.path.isdir(backends_root):
    backend_modules = [os.path.splitext(file)[0] for file in os.listdir(backends_root)
                       if os.path.isfile(os.path.join(backends_root, file)) and file.endswith("_api.py")]
    for backend_module in backend_modules:
        try:
            module = importlib.import_module(f"backends.{backend_module}")
            backend_subclasses = inspect.getmembers(module, predicate=is_backend)
            for name, backend_cls in backend_subclasses:
                _loaded_backends.append(backend_cls())
        except Exception as e:
            print(e)
            print(f"Cannot load 'backends.{backend_module}'."
                  f" Please make sure that the file exists.", file=sys.stderr)
if _loaded_backends:
    print("Loaded backends:", ",".join([str(b) for b in _loaded_backends]))
else:
    print("No backends found. Only programmatic backends possible.", file=sys.stderr)


def lookup_by_model_name(remote_model_name: str) -> Backend:
    """
    :param remote_model_name: the model name for which a supporting backend has to be found
    :return: first backend found that supports the model; otherwise None
    """
    for backend in _loaded_backends:
        if backend.supports(remote_model_name):
            return backend
    return None


def configure(fn_apply: Callable[[Backend], None]):
    """
    :param fn_apply: function to apply on each loaded backend
    """
    for backend in _loaded_backends:
        fn_apply(backend)
