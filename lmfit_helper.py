
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

try:
    from contours.core import numpy_formatter
    from contours.quad import QuadContourGenerator
    import shapely
except ImportError:
    print('contours and/or shapely not installed, please do it by running\npip install contours shapely')
    raise


def fit_plots(x, y, minimizer, minimizer_result, residual,
              model, title='',
              contour_level=0.6827, cmap=plt.cm.coolwarm,
              xlabel='x', ylabel='y', xlim=None, ylim=None):

    params = minimizer_result.params

    ci, trace = lmfit.conf_interval(minimizer, minimizer_result, trace=True)
    param_names = list(params.keys())

    figs = []

    ##################################
    # Figure 1: Fit Parameters as text
    ##################################

    fig = plt.figure()

    s = '%s\n\n' % model
    if title:
        s += title + '\n'
    for ndx, k in enumerate(param_names):
        s += '%s: %.3f ± %.3f' % (k, minimizer_result.params[k].value, minimizer_result.params[k].stderr)
        s += '\n'

    plt.text(0.5, 0.5, s, fontsize=12, ha='center', va='center')
    plt.axis('off')

    figs.append(fig)

    #############################
    # Figure 2: Fit and residuals
    #############################

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(x, y, '.')
    xs = np.linspace(x[0], x[-1])
    ax1.plot(xs, residual(minimizer_result.params, xs), '-')
    ax1.set_ylabel(ylabel)
    if ylim:
        ax1.set_ylim(ylim)

    r = residual(minimizer_result.params, x, data=y)
    ax2.plot(x, r)
    ax2.axhline(y=0, color='k', linestyle=':')
    mx = np.max(np.abs(r))
    ax2.set_ylim([-mx, mx])
    ax2.set_ylabel('R')
    ax2.set_xlabel(xlabel)

    if xlim:
        ax2.set_xlim(xlim)

    figs.append(fig)

    #############################
    # Figure 3: Probability plots
    #############################

    contours = {}

    fig, axs = plt.subplots(len(param_names), len(param_names))

    for ndx1 in range(len(param_names)):
        for ndx2 in range(len(param_names)):
            ax = axs[ndx2][ndx1]

            if ndx1 > ndx2:
                ax.set_axis_off()
                continue

            if ndx1 == ndx2:
                x = trace[param_names[ndx1]][param_names[ndx1]]
                y = trace[param_names[ndx1]]['prob']

                t, s = np.unique(x, True)
                f = interp1d(t, y[s], 'slinear')
                xn = np.linspace(x.min(), x.max(), 50)
                ax.plot(xn, f(xn), 'g', lw=1)

                contours[ndx1] = (x, y)

            else:
                x, y, m = lmfit.conf_interval2d(minimizer, minimizer_result, param_names[ndx1], param_names[ndx2], 20, 20)
                ax.contourf(x, y, m, np.linspace(0, 1, 10), cmap=cmap)

                ch = QuadContourGenerator.from_rectilinear(x, y, m, numpy_formatter)

                contours[(ndx1, ndx2)] = ch.contour(contour_level)

            if ndx1 == 0:
                if ndx2 > 0:
                    ax.set_ylabel(param_names[ndx2])
                else:
                    ax.set_ylabel('prob')
            else:
                ax.set_yticks([])

            if ndx2 == len(param_names) - 1:
                ax.set_xlabel(param_names[ndx1])
            else:
                ax.set_xticks([])

    figs.append(fig)

    return figs, contours


def plot_contours(param_names, grouped_contours, labels, cmap=plt.cm.tab10):

    fig, axs = plt.subplots(len(param_names), len(param_names))

    for k, multiple_contours in grouped_contours.items():

        color = cmap(labels.index(k))

        for contours in multiple_contours:

            for ndx1 in range(len(param_names)):
                for ndx2 in range(len(param_names)):

                    ax = axs[ndx2][ndx1]

                    if ndx1 > ndx2:
                        ax.set_axis_off()
                        continue

                    if ndx1 == ndx2:
                        x, y = contours[ndx1]

                        t, s = np.unique(x, True)
                        f = interp1d(t, y[s], 'slinear')
                        xn = np.linspace(x.min(), x.max(), 50)
                        ax.plot(xn, f(xn), color=color, lw=1, alpha=0.5)

                    else:
                        xy = contours[(ndx1, ndx2)][0]

                        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1, alpha=0.5)

                    if ndx1 == 0:
                        if ndx2 > 0:
                            ax.set_ylabel(param_names[ndx2])
                        else:
                            ax.set_ylabel('prob')
                    else:
                        ax.set_yticks([])

                    if ndx2 == len(param_names) - 1:
                        ax.set_xlabel(param_names[ndx1])
                    else:
                        ax.set_xticks([])

    ax = axs[0][2]

    for ndx, label in enumerate(labels):
        ax.plot([0], [ndx * .5], color=cmap(ndx), label=label)

    ax.legend()

    return fig


if __name__ == '__main__':

    import numpy as np

    def residual(par, x, data=None):
        a = par['a']
        b = par['b']
        c = par['c']

        model = a - c * np.exp(-x / b)
        if data is None:
            return model

        return model - data

    # One
    fit_params = lmfit.Parameters()
    fit_params.add_many(('a', .5), ('b', .5), ('c', .5))

    x = np.linspace(0, 10)
    y = residual(dict(a=2, b=1, c=.3), x)
    y = np.random.normal(y, 0.01)

    mini = lmfit.Minimizer(residual, fit_params, nan_policy='propagate',
                           fcn_args=(x,), fcn_kws={'data': y})

    out = mini.leastsq()

    figs, contour_lines = fit_plots(x, y, mini, out, residual, model='a - c * exp(-x / b)')

    # Multiple

    grouped_contours = {}

    uncertainties = dict(large=0.05, medium=0.01, small=0.005)

    for unc_name, unc_value in uncertainties.items():

        multiple_contours = []
        for n in range(5):
            x = np.linspace(0, 10)
            y = residual(dict(a=2, b=1, c=.3), x)
            y = np.random.normal(y, unc_value)

            fit_params = lmfit.Parameters()
            fit_params.add_many(('a', .5), ('b', .5), ('c', .5))

            mini = lmfit.Minimizer(residual, fit_params, nan_policy='propagate',
                                   fcn_args=(x,), fcn_kws={'data': y})

            out = mini.leastsq()

            other_figs, contours = fit_plots(x, y, mini, out, residual, model='a - c * exp(-x / b)')
            for fig in other_figs:
                plt.close(fig)

            multiple_contours.append(contours)

        grouped_contours[unc_name] = multiple_contours

    fig = plot_contours(('a', 'b', 'c'), grouped_contours, list(uncertainties.keys()))
    plt.show()
