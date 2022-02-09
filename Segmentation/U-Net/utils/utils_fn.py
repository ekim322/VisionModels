def check_config_key(cfg, key):
    """
    Return True if valid value exists for config[key]
    """
    if key in cfg:
        if cfg[key] is not None:
            return True
    return False