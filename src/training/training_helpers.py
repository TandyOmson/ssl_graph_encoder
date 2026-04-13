""" Common functions between pretraining and finetuning
"""
import importlib
import inspect

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def get_all_init_params(cls):
    """
    Will get all parameters accepted by __init__ of a class, including those inherited from parent classes.
    If any __init__ accepts **kwargs, return None (means: accept everything).
    """
    accepted = set()

    for base in cls.__mro__:
        if base is object:
            continue

        if "__init__" not in base.__dict__:
            continue

        sig = inspect.signature(base.__init__)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # if param.kind == inspect.Parameter.VAR_KEYWORD:
            #     # **kwargs present → no filtering should be applied
            #     return None

            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(name)

    return accepted

def filter_class_config(cls, **config):
    accepted = get_all_init_params(cls)

    if accepted is None:
        # class (or one of its parents) accepts **kwargs
        return dict(config)

    return {k: v for k, v in config.items() if k in accepted}

# CALLBACKS
class Callback:
    """
    Base class for training callbacks.
    Return True from on_epoch_end to signal early stopping.
    """

    def on_epoch_begin(self, epoch):
        return False

    def on_epoch_end(self, epoch, metrics):
        return False

    def on_train_end(self):
        pass

class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        monitor="loss",
        mode="min",      # "min" or "max"
        patience=10,
        min_delta=0.0,
        logger=None,
    ):
        assert mode in ("min", "max")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger

        self.best = None
        self.bad_epochs = 0
        self.stopped_epoch = None

    def _is_improvement(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        else:
            return value > self.best + self.min_delta

    def on_epoch_end(self, epoch, metrics):
        value = metrics[self.monitor]

        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            if self.logger is not None:
                self.logger.info(f"[EarlyStopping] Stopping at epoch {epoch}")
            return True  # signal stop

        return False

class ReportMetricsCallback(Callback):
    def __init__(
        self,
        every_n_epochs,
        keys=None,            # list of metric names, or None = all
        fmt: str = ".4f",
        logger=None,
    ):
        self.every_n_epochs = every_n_epochs
        self.keys = keys
        self.logger = logger
        self.fmt = fmt

    def on_epoch_end(self, epoch, metrics):
        if epoch % self.every_n_epochs != 0:
            return False

        if self.keys is None:
            items = metrics.items()
        else:
            items = ((k, metrics[k]) for k in self.keys)

        msg = ", ".join(f"{k}={v:{self.fmt}}" for k, v in items)
        self.logger.debug(f"[Epoch {epoch}] {msg}")

        return False