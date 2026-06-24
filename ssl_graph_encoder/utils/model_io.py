""" Save and load functions for pretrained encoder and full model
"""
import torch
import copy
import importlib
import logging
import sys

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def save_pretrained_encoder(path, encoder, config, extra=None):
    payload = {
        "model": "pretrained_encoder",
        "encoder_class_path": config["encoder"]["class_path"],
        "encoder_kwargs": {k: v for k, v in config["encoder"].items() if k != "class_path" and k != "embed_dim"},
        "embed_dim": config["encoder"]["embed_dim"],
        "feat_dim": config["feat_dim"],
        "state_dict": encoder.state_dict(),
        "extra": extra or {},
        "config_snapshot": copy.deepcopy(config),
    }
    torch.save(payload, path)

def load_pretrained_encoder(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)

    encoder_cls = load_class(payload["encoder_class_path"])
    encoder = encoder_cls(
        payload["feat_dim"],
        payload["embed_dim"],
        **payload["encoder_kwargs"]["kwargs"]
    )
    encoder.load_state_dict(payload["state_dict"], strict=True)
    encoder.eval()
    
    return encoder, payload

def save_full_model(path, encoder, prediction_head, config, extra=None):
    payload = {
        "model": "encoder+prediction_head",

        "feat_dim" : config["feat_dim"],
        # encoder
        "encoder": {
            "class_path": config["encoder"]["class_path"],
            "kwargs": {
                k: v for k, v in config["encoder"].items()
                if k != "class_path"
            },
            "state_dict": encoder.state_dict(),
        },

        # prediction head
        "prediction_head": {
            "class_path": config["prediction_head"]["class_path"],
            "kwargs": {
                k: v for k, v in config["prediction_head"].items()
                if k != "class_path"
            },
            "state_dict": prediction_head.state_dict(),
        },
        "extra": extra or {},
        "config_snapshot": copy.deepcopy(config),
    }

    torch.save(payload, path)

def load_full_model(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)

    # rebuild encoder
    enc_cfg = payload["encoder"]
    encoder_cls = load_class(enc_cfg["class_path"])
    encoder = encoder_cls(
        payload["feat_dim"],
        enc_cfg["kwargs"]["embed_dim"],
        **enc_cfg["kwargs"],
    )
    encoder.load_state_dict(enc_cfg["state_dict"], strict=True)
    encoder.eval()

    # rebuild regression head
    head_cfg = payload["prediction_head"]
    head_cls = load_class(head_cfg["class_path"])

    out_dim = getattr(encoder, "out_dim", enc_cfg["kwargs"]["embed_dim"])
    head = head_cls(out_dim, **head_cfg["kwargs"])
    head.load_state_dict(head_cfg["state_dict"], strict=True)
    head.eval()

    return encoder, head, payload

def setup_logging(log_dir):
    """ configure logging
    """
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    
    try:
        log_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise Exception(f"Log directory {log_dir} already exists. Exiting...")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout (debug level) 
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    log.addHandler(console)

    # run log 
    run_handler = logging.FileHandler(log_dir / "run.log", mode="w")
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(formatter)
    log.addHandler(run_handler)

    # detail log 
    detail_handler = logging.FileHandler(log_dir / "detail.log", mode="w")
    detail_handler.setLevel(logging.DEBUG)
    detail_handler.setFormatter(formatter)
    log.addHandler(detail_handler)

    # pretrain and finetune loggers
    pretrain_logger = logging.getLogger("pretrain")
    pretrain_logger.setLevel(logging.INFO)

    data_handler = logging.FileHandler(log_dir / "pretrain.dat", mode="w")
    data_handler.setFormatter(logging.Formatter("%(message)s"))

    pretrain_logger.addHandler(data_handler)
    pretrain_logger.propagate = False # prevent duplication to root logs

    finetune_logger = logging.getLogger("finetune")
    finetune_logger.setLevel(logging.INFO)

    data_handler = logging.FileHandler(log_dir / "finetune.dat", mode="w")
    data_handler.setFormatter(logging.Formatter("%(message)s"))

    finetune_logger.addHandler(data_handler)
    finetune_logger.propagate = False # prevent duplication to root logs

    return