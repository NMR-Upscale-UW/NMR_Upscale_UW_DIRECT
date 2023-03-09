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

name2params = {
    'mlp': DotDict({
        'input_dim': 5500,
        'output_dim': 5500,
        'hidden_dims': [1024, 128, 512, 512, 4000],
        'p_drop': 0.5,
        'lr': 0.001,
    }),
    'cnn': DotDict({
        'input_dim': 1,
        'output_dim': 1,
        'hidden_dims': [128, 128, 128, 128],
        'kernel_size': 3,
        'p_drop': 0.1,
        'lr': 0.001,
    }),

}