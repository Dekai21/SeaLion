
def get_seg_class(dataset: str, cate: str)-> dict:
    if dataset.lower() == "shapenet" or dataset.lower() == "shapenetpart":
        _seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        _seg_classes_lower = {}
        for seg in _seg_classes.keys():
            _seg_classes_lower[seg.lower()] = _seg_classes[seg]
        _seg_classes.update(_seg_classes_lower)
    elif dataset.lower() == "partnet":
        _seg_nums = {'chair': 6, 'bed': 4, 'bottle': 6, 'display': 3, 'faucet': 8, 'laptop': 3, 'table': 14}
        _seg_classes = {}
        for _cate in _seg_nums.keys():
            _seg_classes[_cate] = [i for i in range(_seg_nums[_cate])]
    elif dataset.lower() == "intra":
        _seg_classes = {'aneurysm': [0, 1]}
    elif dataset.lower() == "ict":
        _seg_classes = {'c_clamp': [0, 1, 2]}
    else:
        raise NotImplementedError
    seg_classes = {}
    seg_classes[cate] = [i - min(_seg_classes[cate]) for i in _seg_classes[cate]]
    return seg_classes


def get_seg_offset(dataset: str, cate: str)-> int:
    if dataset.lower() == "shapenet" or dataset.lower() == "shapenetpart":
        _seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        _seg_classes_lower = {}
        for seg in _seg_classes.keys():
            _seg_classes_lower[seg.lower()] = _seg_classes[seg]
        _seg_classes.update(_seg_classes_lower)
    elif dataset.lower() == "partnet":
        _seg_nums = {'chair': 6, 'bed': 4, 'bottle': 6, 'display': 3, 'faucet': 8, 'laptop': 3, 'table': 14}
        _seg_classes = {}
        for cate in _seg_nums.keys():
            _seg_classes[cate] = [i for i in range(_seg_nums[cate])]
    elif dataset.lower() == "intra":
        _seg_classes = {'aneurysm': [0, 1]}
    elif dataset.lower() == "ict":
        _seg_classes = {'c_clamp': [0, 1, 2]}
    else:
        raise NotImplementedError
    return min(_seg_classes[cate])