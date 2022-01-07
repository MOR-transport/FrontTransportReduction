import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
import os
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

pic_dir = "../imgs"
plt.close("all")

def show_animation(q, Xgrid=None, cycles=1, frequency = 1, figure_number = None):
    ntime = q.shape[-1]

    if figure_number == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(num=figure_number)

    if len(Xgrid) == 1:
        line, = ax.plot(Xgrid[0], q[..., 0])
        plt.ylim([np.min(q), np.max(q)])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q(x,t)$")
        for t in range(0, cycles * ntime, frequency):
            line.set_data(Xgrid[0],q[..., t % ntime].ravel())
            # fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)


    else:

        h = ax.pcolormesh(Xgrid[0], Xgrid[1], q[..., 0])
        h.set_clim(np.min(q), np.max(q))
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        for t in range(0, cycles * ntime, frequency):
            h.set_array(q[:-1, :-1, t % ntime].ravel())
            # fig.savefig("../imgs/FTR_6modes_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)


def show_samplepoints(snapshot, Xgrid, sampleMeshIndices, figure = None):
    if figure == None:
        fig, ax = plt.subplots()
    else:
        fig = figure
        ax = fig.gca()

    h = ax.pcolormesh(Xgrid[0], Xgrid[1], snapshot)
    Xvec, Yvec = Xgrid[0].flatten(), Xgrid[1].flatten()
    Xsample, Ysample = np.take(Xvec, sampleMeshIndices), np.take(Yvec, sampleMeshIndices)
    ax.scatter(Xsample,Ysample,color = 'red')
    h.set_clim(np.min(snapshot),np.max(snapshot))
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([min(Xvec), max(Xvec)])
    ax.set_ylim([min(Yvec), max(Yvec)])
    ax.set_title("sample points")
    return fig
    #ax.set_xlabel(r"$x$")
    #ax.set_ylabel(r"$y$")

def levelset_surface_plot(q,phi, X, figure_number=None):

    fig = plt.figure(figure_number)
    ax = fig.gca(projection='3d')
    Z2 = np.zeros_like(X[0])
    Z3 = q
    # Shade data, creating an rgb array.
    plt.contourf(X[0], X[1], Z2, alpha=0.8)
    plt.contourf(X[0], X[1], Z3, zdir='z', offset=-2)
    ax.plot_surface(X[0], X[1], -0.1*phi, rstride=1, cstride=1, alpha=0.2, antialiased=False, cmap="bwr", linewidth=0)
    ax.set_zticks([0])
    #ax.set_xticks([])
    #ax.set_yticks([])

    ax.text(0, 1, 0.8, r"$\phi(\mathbf{x},t)$")
    ax.text(-0.4, 0.2, -2, r"$q(\mathbf{x},t)$")
    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(0, 1)
    ax.set_zlim3d(-2, 0.5)
    plt.show()
    return fig,ax



def save_fig(filepath, figure=None, **kwargs ):
    import tikzplotlib
    import os

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=600, transparent=True)
    tikzplotlib.save(
        figure = figure,
        filepath=fpath+".tex",
        axis_height = '\\figureheight',
        axis_width = '\\figurewidth',
        override_externals = True,
        **kwargs
    )

def autolabel(rects, ax, fmt ):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt%(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')