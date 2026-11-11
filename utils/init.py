def filter_opt(opt, prefix):
    result = {}
    for key, val in opt.items():
        if key.startswith(prefix + '.'):
            new_key = key[len(prefix) + 1:]
            result[new_key] = val
    return result
