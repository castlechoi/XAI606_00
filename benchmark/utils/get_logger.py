def get_pretrain_model_name(cfg):
    paradigm = cfg.trainer.paradigm
    if hasattr(cfg.dataset, "pre_extract"):
        pretrain_model_name = cfg.dataset.pre_extract.feature_extractor.pretrain.name
    elif hasattr(cfg.model, "feature_extractor"):
        pretrain_model_name = cfg.model.feature_extractor.pretrain.name
    else:
        raise NotImplementedError
    return pretrain_model_name


def get_logger(cfg):
    return None
