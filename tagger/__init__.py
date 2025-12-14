import logging
import os
import sys

from configparser import ConfigParser


config = ConfigParser()
# get config from file
config.read('tagger.ini')

# # get config from environment
# if not config.has_section('BOTZEN'):
#     config.add_section('BOTZEN')
# for key, val in os.environ.items():
#     if key.lower().startswith('botzen_'):
#         # standard interpolation may fail with certain env-values
#         try:
#             config.set('BOTZEN', key[7:], val)
#         except ValueError:
#             print("failed to use config value for %s" % (str({key: val})),
#                   file=sys.stderr)
#             exit(1)
# 
# # dict of env-vars
# CONFIG_ENV_DEFAULTS = dict(config.items('BOTZEN'))
# # name of config section to look for task specific configuration
# CONFIG_TASK = config.get('DEFAULT', 'task', vars=CONFIG_ENV_DEFAULTS,
#                          fallback='empirist')

APP_LOGGER_NAME = "tagger"

logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt='%H:%M:%S',
    format="%(asctime)s %(filename)14s,%(funcName)8s(),ln%(lineno) 3s: %(message)s",
    level=logging.DEBUG)
logger.debug("config read, logging started...")


def setup_applevel_logger(logger_name: str = APP_LOGGER_NAME, fn: str = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if fn:
        fh = logging.FileHandler(fn)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)
