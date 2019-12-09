import numpy as np


def auto_pate(method):
    """自动添加括号"""
    method = str.strip(method)
    if method[-1]!=')':
        if '(' not in method:
            method = method+'()'
        else:
            method =  method+')'
    return method


def back_args_str(*args, **kwargs):
    largs = [f"'{str(a)}'" if isinstance(a, str) else str(a) for a in args]
    kw = [str(k) + '=' + ("'" + str(v) + "'" if isinstance(v, str) else str(v)) for k, v in kwargs.items()]
    largs.extend(kw)
    return ','.join(largs)


def mad_based_outlier(points, thresh=3.5):
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    # diff = np.sum((points - median) ** 2, axis=-1)
    # diff = np.sqrt(diff)
    diff = np.abs(points-median)
    med_abs_deviation = np.median(diff, axis=0)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    result = modified_z_score > thresh
    return result.squeeze().tolist()

