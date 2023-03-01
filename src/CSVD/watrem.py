import hlsvdpro
import numpy as np


def watrem(data, dt, n, f):
    """
    `watrem` takes a time series, a sampling interval, the number of singular values to use, and a frequency cutoff, and
    returns the time series with the low frequency components removed.

    :param data: the data to be filtered
    :param dt: the dwell time of the data in seconds
    :param n: number of singular values to use
    :param f: the frequency cutoff for the HLSVD
    :return: The residuals of the data after the removal of the HLSVD components.
    """
    npts = len(data)
    dwell = dt
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    idx = np.where((f[0] < result[2]) & (result[2] < f[1]))
    result = (len(idx), result[1], result[2][idx], result[3][idx], result[4][idx], result[5][idx])
    fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)
    return data - fid

def watrem_batch(dataset, dt, n, f):
    dataset_ = np.zeros_like(dataset)
    for idx in range(len(dataset[0])):
        dataset_[:,idx] = watrem(dataset[:,idx],dt, n, f)
        if idx % 100 == 0:
            print(str(idx))
    return dataset_