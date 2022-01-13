import numpy as np
import torch
#from prefetch_generator import background


def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    
    Y, X = np.meshgrid(y, x)
    return X, Y


def phi(X, Y, t, R):
    t = [t] if isinstance(t, float) else t
    R = [R] if isinstance(R, float) else R
    
    t = np.expand_dims(t, [1, 2])
    R = np.expand_dims(R, [1, 2])
    X = np.expand_dims(X, [0])
    Y = np.expand_dims(Y, [0])
    
    x0 = np.max(X) * (1 + np.cos(2 * np.pi * t) / 2) / 2
    y0 = np.max(Y) * (1 + np.sin(2 * np.pi * t) / 2) / 2
    
    return np.sqrt((X - x0)**2 + (Y - y0)**2) - R


def f(phi, lam):
    lam = [lam] if isinstance(lam, float) else lam
    lam = np.expand_dims(lam, [1, 2])
    return (np.tanh(phi / lam) + 1) / 2


def generate_data(X, Y, t, R, lam, to_torch=False, type="moving_disc", device='cpu', dtype=torch.float32):
    if type == "moving_disc":
        phi_field = phi(X, Y, t, R)
    elif type == "pear":
        L = [4, 4]
        N = [200, 200 ]
        Ntime = 50
        x = np.linspace(-L[0], L[0], N[0])
        y = np.linspace(-L[1], L[1], N[1])
        dx = x[2]-x[1]
        dy = y[2]-y[1]
        t = np.linspace(0,2,Ntime)
        T, Y, X = np.meshgrid(t, y, x,  indexing='ij')
        phi_field = 0.4*T-0.5*(X+2+0.5*T)*X**2*(X-2-0.5*T)-Y**2-0.5
        lam = 2*min(dx, dy)
    else:
        L = [10, 10]
        N = [2**8, 2**8]
        Nt = 51
        x = np.linspace(0, L[0], N[0])
        y = np.linspace(0, L[1], N[1])
        dx = x[2] - x[1]
        dy = y[2] - y[1]
        lam = 2 * min(dx, dy)
        t = np.linspace(0, 0.5, Nt)
        T, X, Y = np.meshgrid(t, y, x, indexing='ij')

        xmid = np.asarray([[0.75, 0.25, 0.5], # x coordinates
                [0.35, 0.5, 0.76]]) # y coordinates
        xmid = xmid * min(L)
        Amplitude = [1, 1.4, 1.2]
        sigma = [0.1, 0.3, 0.5]
        phi_field = np.ones_like(X)
        for i in range(3):
            R = ((X - xmid[0, i])**2 + (Y - xmid[1, i])**2)
            phi_field = phi_field - Amplitude[i] * np.exp(-sigma[i] * R)
        phi_field = phi_field - T
    q = np.expand_dims(f(phi_field, lam), 1)
    q = torch.tensor(q, dtype=dtype, device=device) if to_torch else q
    return q, phi_field

#@background(max_prefetch=3)
def batch_generator(params, X, Y, batch_size, device, return_t=True):
    step = 0
    while True:
        batch_idx = step % (params.shape[0] - batch_size)
        batch_params = params[batch_idx: batch_idx+batch_size, ...]
        batch = generate_data(X, Y, batch_params[:, 0], batch_params[:, 1], batch_params[:, 2], to_torch=True, device=device)
        step += 1
        if return_t:
            yield batch, batch_params[:, [0]]
        else:
            yield batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    Nx, Ny = 2**7, 2**7
    Lx, Ly = 1, 1
    R = 0.3

    X, Y = make_grid(Nx, Ny, Lx, Ly)
    q, _ = generate_data(X, Y, 1, R, 1 , to_torch=False, type="dw", device='cpu')

    for i in range(0,np.size(q,0),2):
        plt.pcolormesh(np.squeeze(q[i,...]))
        plt.draw()
        plt.pause(0.01)


