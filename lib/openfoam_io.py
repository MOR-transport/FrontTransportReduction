"""
Reads and interpolates open foam time series data on uniform mesh
=================================================================
"""

# import readmesh function from fluidfoam package

import os
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from matplotlib.pyplot import figure, draw, pause

def read_data(fname):
    import scipy.io
    data = scipy.io.loadmat(fname)
    return data["data"], data["xgrid"], data["ygrid"]

def read_openfoam2D(sol, qty_name, ngridx=2**10, ngridy=2**6, clean='Normalize',min_val=0.0, input_uniform_mesh=False, mirrow_data=False):
    from fluidfoam import readvector, readscalar
    from fluidfoam import readmesh
    # read mesh
    x, y, z = readmesh(sol)

    if input_uniform_mesh:
        xi = np.sort(np.unique(x))
        if len(np.unique(y))==1: # in martins data y<->z
            y = z
        yi = np.sort(np.unique(y))
    else:
        # Interpolation grid dimensions
        xinterpmin = np.min(x)
        xinterpmax = np.max(x)
        yinterpmin = np.min(y)
        yinterpmax = np.max(y)

        # Interpolation grid
        xi = np.linspace(xinterpmin, xinterpmax, ngridx)
        yi = np.linspace(yinterpmin, yinterpmax-yinterpmin, ngridy)

    # Structured grid creation
    xinterp, yinterp = np.meshgrid(xi, yi)

    # find time series files
    dir_list = os.listdir(sol)
    time_list = []
    for directory in dir_list:
        try:
            float(directory)
            time_list.append(directory)
        except:
            pass
    time_list.sort(key=float)
    time_list=np.array(time_list)

    # read data of all files into data list
    data_list = []
    time = []
    for timename in time_list:
        if not os.path.exists(sol+"/"+timename+"/"+qty_name):
            continue
        try:
            phi_vec = readscalar(sol, timename, qty_name)
            n_components=1
            phi_vec = np.reshape(phi_vec,[1,*phi_vec.shape])
        except:
            phi_vec = readvector(sol, timename, qty_name)
            n_components = phi_vec.shape[0]

        # Interpolation of scalar fields and vector field components
        if mirrow_data:
            phi_i = np.zeros([n_components, xinterp.shape[0] * 2, xinterp.shape[1]])
        else:
            phi_i = np.zeros([n_components, xinterp.shape[0], xinterp.shape[1]])
        for nc in range(n_components):
            if np.size(phi_vec, -1) == 1:  # open foam sets only one value for the initial condition as it seems
                phi_dat = np.ones_like(x) * phi_vec[nc]
            else:
                phi_dat = phi_vec[nc]
            phi_temp = griddata((x, y), phi_dat, (xinterp, yinterp), method='linear')
            if mirrow_data:
                phi_i[nc] = np.append(np.flipud(phi_temp), phi_temp, 0) # mirrow data
            else:
                phi_i[nc] = phi_temp
        # clean data if nessesary
        if clean == "CutAndNormalize": # cuts outlayers
            phi_i = np.where(phi_i > min_val, phi_i, min_val)
            phi_i = phi_i - np.min(phi_i.flatten())
            phi_i = phi_i / np.max(phi_i.flatten())
        elif clean=="Normalize": # normalizes to 1
            phi_i = phi_i - np.min(phi_i.flatten())
            phi_i = phi_i/np.max(phi_i.flatten())
        else: # do nothing
            pass
        # put data in list
        data_list.append(phi_i)
        time.append(float(timename))

    if mirrow_data:
        # mirrow data on the symmetrie axis
        yinterp = np.append(-np.flipud(yinterp),yinterp,0)
        xinterp = np.append(xinterp,xinterp,0)

    data = np.asarray(data_list)

    return data, xinterp, yinterp, time

###############################################################################
# Example for plotting
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.io import savemat


    qty_name="U"
    crop = False
    cleaning_method=""#"CutAndNormalize"; min_val=0.03 # Choose between [CutAndNormalize | Normalize],
    # CutAndNormalize: data points below min_val will be cutted befor normalizing
    #sol = '/home/phil/paper/06_collaboration/02_MartinIsozm/XX_twoFallParticles/'
    sol = '/home/phil/tubCloud/FTR/04_FlamePinchOff/openfoam/'
    data, xinterp, yinterp, time = read_openfoam2D(sol, qty_name, ngridx=2 ** 10, ngridy=2 ** 6, clean=cleaning_method, min_val=0.03, input_uniform_mesh=True)

    # save data to folder
    data_folder = os.path.split(os.path.dirname(sol))[0]

    fname = qty_name+"_"+cleaning_method+"_" if cleaning_method else qty_name+"_"
    if crop:
        xcut,ycut = [np.zeros([xinterp.shape[0],430])]*2
        data ,xinterp, yinterp= data[...,170:int(600)], xinterp[:,170:600]-xinterp[1,170], yinterp[:,170:600]
        fname = fname + "smalldomain.mat"
    else:
        #fname = fname+"fulldomain.mat"
        fname = qty_name + ".mat"

    mydata = {"data" : data, "xgrid" : xinterp, "ygrid" : yinterp, "time": time}
    savemat(data_folder+"/"+fname, mydata)
    # Define plot parameters
    fig = plt.figure()
    ax = fig.gca()
    d=0.005
    plt.xlabel(r'$x/D$')
    plt.ylabel(r'$y/D$')
    plt.title(qty_name)
    h = ax.pcolormesh(xinterp/d, yinterp/d, data[10,0,::],shading='gouraud')
    ax.axis("image")
    # Plots the contour of sediment concentration
    # for i,img in enumerate(data):
    #     h.set_array(img.ravel())
    #     draw(), pause(1e-3)
    #     print("saving file %3.3d"%i)
    #     #fig.savefig("imgs/flame/flame_%3.3d.png"%i)