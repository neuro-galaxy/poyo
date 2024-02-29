##################################################################
# # disable warnings
# import warnings
# warnings.simplefilter("ignore", UserWarning)

# # disable matplotlib output
# import matplotlib
# matplotlib.use('Agg')

# logging with rich
import logging as logging_

from rich.logging import RichHandler

logging_.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True, markup=True, omit_repeated_times=True, show_path=False
        )
    ],
)
log = logging_.getLogger("rich")

# install pretty print in the REP
# from rich import pretty, traceback
# pretty.install(indent_guides=True)  # install in the REPL
# traceback.install(indent_guides=True)  # install in the REPL
##################################################################
from functools import lru_cache

from absl import flags

flags.DEFINE_enum(
    "logging",
    "info",
    ["debug", "info", "warn", "error", "critical"],
    "Set logging level. (default: info)",
)


class logging:
    r"""Provides easy access to logging level through :attr:`debug` and :attr:`info`,
    and restricts logging to master rank only.

    - Sets the level of all loggers to :obj:`level`.
    - Sets up rich loggers and installs pretty print in the REPL from the :mod:`rich` package.

    Args:
        level (int, Optional): Logging level. (Default: :obj:`logging.INFO`)
    """
    level = logging_.INFO
    rank = None

    def __init__(self, *, header: str = None, header_color: str = "black"):
        if header is not None:
            self.header = "[bold {}]{}[/bold {}] ".format(
                header_color, header, header_color
            )
        else:
            self.header = ""
        # if self.is_master_rank:
        #     log.info("[bold pink](੭｡╹▿╹｡)੭[/bold pink] Poyo!")

    @staticmethod
    def getLogger(header: str = None, header_color: str = "black"):
        return logging(header=header, header_color=header_color)

    @classmethod
    def init_logger(cls, level: str = "info", rank: int = 0):
        # logging_.getLogger().handlers.clear()
        logging_.getLogger().addHandler(
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                omit_repeated_times=True,
                show_path=False,
                log_time_format="[%X]",
            )
        )
        level = getattr(logging_, level.upper())

        cls.level = level if rank == 0 else "error"
        cls.rank = rank

        log.setLevel(level)

    def info(self, message: str):
        if self.is_enabled_for_info:
            log.info(self.header + message)

    def debug(self, message: str):
        if self.is_enabled_for_debug:
            log.debug(self.header + message)

    def warning(self, message: str):
        if self.is_enabled_for_warning:
            log.warning(self.header + message)

    def error(self, message: str):
        if self.is_enabled_for_error:
            log.error(self.header + message)

    def critical(self, message: str):
        if self.is_enabled_for_critical:
            log.critical(self.header + message)

    @property
    def is_master_rank(self) -> bool:
        r"""Whether this is the master node."""
        return self.rank is None or self.rank == 0

    @property
    @lru_cache(2)
    def is_enabled_for_info(self) -> bool:
        r"""Whether info level is enabled."""
        return log.isEnabledFor(logging_.INFO) and self.is_master_rank

    @property
    @lru_cache(2)
    def is_enabled_for_debug(self) -> bool:
        r"""Whether debug level is enabled."""
        return log.isEnabledFor(logging_.DEBUG) and self.is_master_rank

    @property
    @lru_cache(2)
    def is_enabled_for_warning(self) -> bool:
        r"""Whether warning level is enabled."""
        return log.isEnabledFor(logging_.WARNING) and self.is_master_rank

    @property
    @lru_cache(2)
    def is_enabled_for_error(self) -> bool:
        r"""Whether error level is enabled."""
        return log.isEnabledFor(logging_.ERROR) and self.is_master_rank

    @property
    @lru_cache(2)
    def is_enabled_for_critical(self) -> bool:
        r"""Whether critical level is enabled."""
        return log.isEnabledFor(logging_.CRITICAL) and self.is_master_rank


def set_level_all_loggers(level):
    loggers = [
        logging_.getLogger(name)
        for name in logging_.root.manager.loggerDict
        if name != "rich"
    ]
    for logger in loggers:
        logger.setLevel(level)


set_level_all_loggers(logging_.INFO)