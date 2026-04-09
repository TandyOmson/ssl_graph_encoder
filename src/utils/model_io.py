""" Save and load functions for pretrained encoder and full model
"""
import torch
import copy
import importlib

def load_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def save_pretrained_encoder(path, encoder, config, extra=None):
    payload = {
        "model": "pretrained_encoder",
        "encoder_class_path": config["encoder"]["class_path"],
        "encoder_kwargs": {k: v for k, v in config["encoder"].items() if k != "class_path"},
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
        **payload["encoder_kwargs"]
    )
    encoder.load_state_dict(payload["state_dict"], strict=True)
    encoder.eval()
    
    return encoder, payload

def save_full_model(path, encoder, regression_head, config, extra=None):
    payload = {
        "model": "encoder+regression_head",

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

        # regression head
        "regression_head": {
            "class_path": config["regression_head"]["class_path"],
            "kwargs": {
                k: v for k, v in config["regression_head"].items()
                if k != "class_path"
            },
            "state_dict": regression_head.state_dict(),
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
    head_cfg = payload["regression_head"]
    head_cls = load_class(head_cfg["class_path"])

    out_dim = getattr(encoder, "out_dim", enc_cfg["kwargs"]["embed_dim"])
    head = head_cls(out_dim, **head_cfg["kwargs"])
    head.load_state_dict(head_cfg["state_dict"], strict=True)
    head.eval()

    return encoder, head, payload