import logging
import coloredlogs

# log_config = {
#   'version': 1.0,
#   'formatters': {
#     'colored_console': {
#       '()': 'coloredlogs.ColoredFormatter',
#       'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#       'datefmt': '%Y-%m-%d %H:%M:%S'
#     },
#     'detail': {
#       'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#       'datefmt': "%Y-%m-%d %H:%M:%S"
#     },
#     'simple': {
#       'format': '%(name)s - %(levelname)s - %(message)s',
#     },
#   },
#   'handlers': {
#     'console': {
#       'class': 'logging.StreamHandler',
#       'level': 'INFO',
#       'formatter': 'colored_console'
#     },
#   },
#   # 'loggers': {
#   #   'crawler': {
#   #     'handlers': ['console'],
#   #     'level': 'DEBUG',
#   #   },
#   #   'other': {
#   #     'handlers': ['console'],
#   #     'level': 'INFO',
#   #   },
#   #   'storage': {
#   #     'handlers': ['console'],
#   #     'level': 'INFO',
#   #   }
#   # }
# }

__formatter = {
    '()': 'coloredlogs.ColoredFormatter',
    'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    'datefmt': '%m-%d %H:%M:%S'
}
logging.basicConfig(format=str(__formatter))

__field_style = dict(
    asctime=dict(color='white'),
    hostname=dict(color='green'),
    levelname=dict(color='yellow', bold=True),
    programname=dict(color='cyan'),
    name=dict(color='cyan')
)

__loggers = {}

logging.getLogger().setLevel(level=logging.WARN)


def get_logger(name) -> logging.Logger:
    if name not in __loggers:
        logger = logging.getLogger(name)
        coloredlogs.install(level=logging.DEBUG, logger=logger,
                            field_styles=__field_style)
        __loggers[name] = logger
    return __loggers[name]
