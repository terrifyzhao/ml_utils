import numpy as np


def back_args_str(*args, **kwargs):
    largs = [f"'{str(a)}'" if isinstance(a, str) else str(a) for a in args]
    kw = [str(k) + '=' + ("'" + str(v) + "'" if isinstance(v, str) else str(v)) for k, v in kwargs.items()]
    largs.extend(kw)
    return ','.join(largs)


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    # diff = np.sum((points - median) ** 2, axis=-1)
    # diff = np.sqrt(diff)
    diff = np.abs(points-median)
    med_abs_deviation = np.median(diff, axis=0)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

