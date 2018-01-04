
import collections
import pathlib

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
from PIL import Image


def generate_lags(length, num, mx=None):
    mx = min(length // 2, int(mx))
    lags = np.logspace(0, np.log10(mx), num)
    lags = np.rint(lags).astype(np.int64)
    return np.unique(lags)


@nb.njit()
def acf(ndx_lags, data):
    n = data.size
    mean = np.sum(data) / n
    mean2 = mean ** 2

    acf_coeffs = np.zeros(ndx_lags.shape)
    for ndx in range(ndx_lags.size):
        h = ndx_lags[ndx]
        acf_coeffs[ndx] = ( (data[:(n - h)] - mean) * (data[h:] - mean) ).mean() / mean2

    return acf_coeffs


def fit(lags, ac):

    p = lmfit.Parameters()
    p.add_many(('N', 1 / ac[:10].mean(), True, 0),
               ('td', lags[lags.size//2], True, 0),
               #('td', 2.5, False),
               ('a', 6., True, 0),
               #('a', 6., False),
               ('b', 0))

    def residual(p):
       N = p['N']
       nt = lags / p['td']
       a2 = p['a'] ** 2
       b = p['b']
       return (N * (1 + nt) * np.sqrt(1 + nt / a2))**(-1) + b - ac

    # create Minimizer
    mini = lmfit.Minimizer(residual, p)

    return mini.minimize(method='leastsq')


def load_oif(oifpath, ch=1):
    oifpath = pathlib.Path(oifpath)
    datafile = oifpath.with_suffix('.oif.files') / ('s_C%03d.tif' % ch)
    metadatafile = oifpath.with_suffix('.oif.files') / ('s_C%03d.pty' % ch)
    im = Image.open(datafile)

    with metadatafile.open('r', encoding='utf-16', errors='replace') as fi:
        for line in fi:
            if 'Time Per Pixel' in line:
                header, value = line.strip().split('=')
                break
        else:
            raise ValueError('Time per Pixel not found')

    return np.array(im).flatten().astype(np.float32), 0.001 * float(value.strip('"'))


DataRecord = collections.namedtuple('DataRecord', 'mean var length binsize')
FitRecord = collections.namedtuple('FitRecord', 'N td a b chi2 success')

def fitted_to_record(fitted):
    return FitRecord(fitted.params['N'].value, fitted.params['td'].value,
                     fitted.params['a'].value, fitted.params['b'].value,
                     fitted.chisqr, fitted.success)


def calculate(data, pixel_time, lags_or_num_lags, max_ndx_lag):
    if isinstance(lags_or_num_lags, int):
        ndx_lags = generate_lags(len(data), lags_or_num_lags, max_ndx_lag)
    else:
        if lags_or_num_lags.dtype == np.int64:
            ndx_lags = lags_or_num_lags
        else:
            ndx_lags = np.rint(lags_or_num_lags / pixel_time).astype(np.int64)

    ac = acf(ndx_lags, data)

    lags = ndx_lags * pixel_time
    fitted = fit(lags, ac)

    ac_stack = np.stack([lags, ac, fitted.residual]).T

    return DataRecord(np.mean(data), np.var(data), len(data), pixel_time), fitted_to_record(fitted), ac_stack


def display(title, data, data_record, fit_record, ac_stack, acs=None):
    print(title)
    print('-' * len(title))

    if data is not None:
        plot_data(data, data_record.binsize)
        plt.show()

    print('Data length: %d' % data_record.length)
    print('Total time: %.3f ms' % (data_record.length * data_record.binsize))

    if data_record is not None:
        print('Mean: %.3f' % data_record.mean)
        print('Var: %.3f' % data_record.var)

    if ac_stack is not None:
        plot_ac(ac_stack, acs)
        plt.show()

    if fit_record is not None:
        print('N: %.2f' % fit_record.N)
        print('td: %.3f ms' % fit_record.td)
        print('a: %.3f' % fit_record.a)
        print('b: %.3f' % fit_record.b)
        print('chi2: %.3f' % fit_record.chi2)
        print('success: %.3f' % fit_record.success)

    print('')

def onefile(oifpath, lags_or_num_lags=50, max_lag=5000, show=False):
    data, pixel_time = load_oif(oifpath)
    data_record, fit_record, ac_stack = calculate(data, pixel_time, lags_or_num_lags, max_lag / pixel_time)

    if show:
        display(str(oifpath), data, data_record, fit_record, ac_stack)

    return data_record, fit_record, ac_stack


def manyfiles(folder, num_lags, max_lag=5000, average=False, recurse=False, show=False):
    folder = pathlib.Path(folder)

    table = []

    if recurse:
        iterp = folder.rglob('*.oif')
    else:
        iterp = folder.glob('*.oif')

    st = []
    binsize = None

    for p in iterp:

        data_record, fit_record, ac_stack = onefile(p, num_lags, max_lag, show)

        if average:
            st.append(ac_stack[:, 1])
            num_lags = ac_stack[:, 0]

        np.savetxt(str(p.with_suffix('.acf.csv')), ac_stack, delimiter=',')

        table.append((str(p), ) + data_record + fit_record)

    if average:
        np.savetxt(str(folder.with_suffix('.avg.acf.csv')), ac_stack, delimiter=',')

        st = np.asarray(st)

        lags = num_lags
        ac = np.mean(st, axis=0)

        fitted = fit(lags, ac)

        data_record = DataRecord(0, 0, 0, 0)
        fit_record = fitted_to_record(fitted)

        if show:
            display('avg ' + str(folder), None, data_record, fit_record, ac_stack, st)

        table.append(('avg ' + str(folder),) + data_record + fit_record)


    return pd.DataFrame(table, columns=['path', 'mean', 'var', 'length', 'binsize',
                                        'N', 'td', 'a', 'b',
                                        'chi2', 'success'])


def oneday(folder, num_lags, max_lag=5000, show=False):
    folder = pathlib.Path(folder)

    subfolders = (p for p in folder.iterdir() if p.is_dir())

    dfs = []
    for s in subfolders:
        dfs.append(repeats(s, num_lags, max_lag, recurse=False, show=show))

    return pd.concat(dfs)


def repeats(folder, num_lags, max_lag=5000, recurse=False, show=False):

    return manyfiles(folder, num_lags, max_lag, average=True, recurse=recurse, show=show)


def plot_data(data, pixel_time):
    plt.plot(np.arange(len(data)) * pixel_time, data)
    plt.ylabel('Intensity')
    plt.xlabel('Time [ms]')


def plot_ac(ac_stack, acs):
    lags = ac_stack[:, 0]
    ac  = ac_stack[:, 1]
    residual = ac_stack[:, 2]

    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    ax1.plot(lags, ac, '.')
    ax1.plot(lags, ac + residual)
    if acs is not None:
        ax1.plot(lags, acs.T, color=(0.5, 0.5, 0.5))

    ax1.set_xticks([])
    ax1.set_xscale('log')
    ax1.set_ylabel('ACF')

    ax2.plot(lags, residual)
    mx = np.max(np.abs(np.min(residual)), np.abs(np.max(residual)))
    ax2.set_ylim([-mx, mx])
    ax2.set_ylabel('R')
    ax2.set_xlabel(r'$\tau \; [ms]$')
