class DotDict(dict):
    """
    Source: https://stackoverflow.com/questions/13520421/recursive-dotdict
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

name2params = { #Use this dictionary to access model's hyperparameters!
    'mlp': DotDict({
        'input_dim': 5500,
        'output_dim': 5500,
        'hidden_dims': (1024, 128, 512, 512, 4000),
        'p_drop': 0.5,
        'lr': 0.001,
    }),
    'cnn': DotDict({
        'input_dim': 1,
        'output_dim': 1,
        # 'hidden_dims': [128, 128, 128, 128],
        'hidden_dims': (64, 32),
        'kernel_size': 7,
        'padding': 3,
        'p_drop': 0.1,
        'lr': 0.001,
    }),
    'conv_vae': DotDict({
        'encoder': {
            'input_dim': 1,
            'output_dim': 2,
            # 'hidden_dims': [128, 128, 128, 128],
            'hidden_dims': (16, 16),
            'kernel_size': 3,
            'padding': 1,
            'p_drop': 0.0,
        },
        'decoder': {
            'input_dim': 1,
            'output_dim': 1,
            # 'hidden_dims': [128, 128, 128, 128],
            'hidden_dims': (16, 16),
            'kernel_size': 3,
            'padding': 1,
            'p_drop': 0.0,
        },
        'lr': 0.001,
    }),
    'transformer': DotDict({
        'input_dim': 1,
        'd_model': 4,
        'nhead': 2,
        'd_hid': 8,
        'nlayers': 1,
        'padding': 3,
        'p_drop': 0.3,
        'lr': 0.001,
    }),

}
