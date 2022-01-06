import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
#from moving_disk_data import batch_generator
from NeuralFTR.utils import save_codeBase, to_torch, to_numpy
import os

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    """
    Trainer for an Autoencoder like NN for 2D data
    """
    def __init__(self,
                 net,
                 X,
                 Y,
                 train_set_params=None,
                 train_set=None,
                 test_set=None,
                 lr=0.001,
                 lr_min=0.0001,
                 smooth_phi=0.0005,
                 sparse_reg=1,
                 sparsity_measure=1,
                 log_folder='./train_results_local/',
                 device=DEVICE,
                 ):
        
        self.net = net.to(device)
        self.X = X
        self.Y = Y
        self.Grid = torch.tensor(np.concatenate([X[None, ...], Y[None, ...]], 0), dtype=torch.float32, device=device)
        self.train_set_params = train_set_params
        self.train_set = train_set
        self.test_set = test_set
        #self.reconstruction_loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.lr_min = lr_min
        self.smooth_phi = smooth_phi
        self.sparse_reg = sparse_reg
        self.sparsity_measure = sparsity_measure
        self.logs_folder = log_folder
        # self.optim = torch.optim.SGD(params=self.net.parameters(), lr=lr, momentum=0.5)
        self.device = device
        
        # Set up a kernel to compute the 2D 2nd order central finite difference in along both axes via 2 conv2d channels
        # self.dx = self.X[1, 0] - self.X[0, 0]
        # self.dy = self.Y[0, 1] - self.Y[0, 0]
        self.dx = self.X[0, 1] - self.X[0, 0] if (self.X[0, 1] - self.X[0, 0]) != 0 else self.X[1, 0] - self.X[0, 0]
        self.dy = self.Y[1, 0] - self.Y[0, 0] if (self.Y[1, 0] - self.Y[0, 0]) != 0 else self.Y[0, 1] - self.Y[0, 0]
        self.D_kernel = torch.zeros([2, 3, 3], device=self.device, dtype=torch.float32)
        self.Dx_kernel = torch.ones([1, 1, 2, 1], device=self.device, dtype=torch.float32) / self.dx
        self.Dy_kernel = torch.ones([1, 1, 1, 2], device=self.device, dtype=torch.float32) / self.dy
        self.pady = torch.nn.ReplicationPad2d(padding=[1, 0, 0, 0])
        self.padx = torch.nn.ReplicationPad2d(padding=[0, 0, 1, 0])
        self.Dx_kernel[0, 0, 0, 0] *= -1
        self.Dy_kernel[0, 0, 0, 0] *= -1
        self.D_kernel[[0, 1], [0, 1], [1, 0]] = -1
        self.D_kernel[[0, 1], [-1, 1], [1, -1]] = 1
        self.D_kernel[0, ...] /= 2 * self.dx
        self.D_kernel[1, ...] /= 2 * self.dy
        self.D_kernel = self.D_kernel[:, None, ...]
        
    def test(self, data, batch_size=None):
        data_len = data.shape[0]
        test_loss = []
        self.net.eval() # set net to evaluation mode
        with torch.no_grad(): # no gradients needed
            if batch_size and batch_size < data_len:
                n_batches = int(np.ceil(data_len / batch_size))
                batch_idx = 0
                for n in range(n_batches):
                    b_size = min(batch_size, data_len-batch_idx)
                    batch_q = data[batch_idx: batch_idx + b_size, ...]
                    batch_idx += b_size
                    test_loss.append(self.get_loss(batch_q, grad=False))
                test_loss = sum(test_loss)/len(test_loss)
            else:
                test_loss = self.get_loss(data, grad=False)
        self.net.train() # set net back to training mode
        return test_loss
    
    def reconstruction_loss(self, q_hat, q):
        reco_loss = (torch.norm((q_hat - q).flatten(1), p='fro') / torch.norm(q.flatten(1), p='fro')) ** 2
        if not reco_loss.isfinite().all():
            debug=1
        return reco_loss
        
    def smoothness_loss(self, field):
        dfield_dx = F.conv2d(self.padx(field), self.Dx_kernel)
        dfield_dy = F.conv2d(self.pady(field), self.Dy_kernel)
        abs_grad_field = (torch.cat([dfield_dx, dfield_dy], 1)**2).sum(1).sqrt()
        # smoothness_loss = ((torch.norm(abs_grad_field, p='fro', dim=(-2, -1)) / torch.norm(field, p='fro'))**2).mean()
        smoothness_loss = (torch.norm(abs_grad_field.flatten(1), p='fro') / torch.norm(field.flatten(1), p='fro')) ** 2
        if not smoothness_loss.isfinite().all():
            debug=1
        return smoothness_loss
    
    def sparsity_loss(self, x):
        return (torch.norm(x.flatten(1), p=self.sparsity_measure) / torch.norm(x.flatten(1), p='fro')) ** 2
    
        
    def get_loss(self, batch, grad=True):
        batch = to_torch(batch, self.device)
        
        def loss_fn(batch_data):
            loss = 0
            q = batch_data
            if self.net.has_bottleneck:
                code, phi, q_hat = self.net(q, return_code=True, return_phi=True)
            else:
                phi, q_hat = self.net(q, apply_f=True, return_phi=True)
            loss += self.reconstruction_loss(q_hat, q)
            if self.smooth_phi:
                if hasattr(self.net, 'decoder') and hasattr(self.net.decoder, 'get_modes'):
                    modes = self.net.decoder.get_modes()[: , None, ...]
                    loss += self.smoothness_loss(modes) * self.smooth_phi
                else:
                    loss += self.smoothness_loss(phi) * self.smooth_phi

            if self.sparse_reg:
                if self.sparsity_measure == 'nuc' or not self.net.has_bottleneck:
                    loss += self.sparsity_loss(phi) * self.sparse_reg
                else:
                    loss += self.sparsity_loss(code) * self.sparse_reg
                    loss += self.declining_code_loss(code) * self.sparse_reg
            return loss
        
        if grad:
            loss = loss_fn(batch)
            return loss
        else:
            with torch.no_grad():
                loss = loss_fn(batch)
                return loss
    
    def resume_training(self, log_base_name):
        same_name = [dir_name for dir_name in os.listdir(self.logs_folder) if log_base_name in dir_name]#[-1]
        # TBD
        
    def get_batch(self, step, batch_size):
        idx = step % self.train_set.shape[0]
        batch_data = self.train_set.roll(-idx, 0)[:batch_size, ...]
        return batch_data
    
    def training(self, batch_size, trainsteps=1e5, test_every=1e3, save_every=5e3, test_batch_size=None, log_base_name='', resume=False):
        log_folder = self.logs_folder + log_base_name + time.strftime("%Y_%m_%d__%H-%M", time.localtime()) + '/'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
        if resume:
            self.resume_training(log_base_name)
        self.tensorboard = SummaryWriter(log_dir=log_folder) if TB_MODE else None
        save_codeBase(os.getcwd(), log_folder)

        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, trainsteps, self.lr_min)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=500, cooldown=1000, verbose=True)
        
        test_loss = torch.zeros(1)
        train_loss_log = []
        test_loss_log = []
        
        # if self.train_set_params is not None:
        #     batch_gen = batch_generator(self.train_set_params, self.X, self.Y, batch_size, device=DEVICE)
        
        
        #train loop
        best_so_far = 1e12
        for step in range(int(trainsteps)):
            
            #learn:
            # if self.train_set_params is not None:
            #     batch = next(batch_gen)
            # else:
            batch = self.get_batch(step, batch_size)
            loss = self.get_loss(batch)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            train_loss_log.append([loss.item(), step])
            if (step + 1) % save_every == 0:
                self.net.save_net_weights(fpath=log_folder + 'net_weights/', fname='step_' + str(step) + '.pt')
            # test/validate:
            if (step + 1) % test_every == 0:
                test_loss = self.test(self.test_set, batch_size=test_batch_size)
                test_loss_log.append([test_loss.item(), step])
                
                reco_error = self.get_reco_error(self.test_set)
                if reco_error < best_so_far:
                    best_so_far = reco_error
                    self.net.save_net_weights(fpath=log_folder + 'net_weights/', fname='best_results.pt')
                    f = open(log_folder + 'net_weights/best_results.txt', 'w')
                    f.write(f"step: {step} ;  Error: {reco_error:.3e}")
                    f.close()
                if self.tensorboard:
                    fig_reco = self.plot_test_idx_reco()
                    fig_modes = self.plot_modes()
                    self.tensorboard.add_figure('reconstruction', fig_reco, global_step=step, close=True)
                    if fig_modes is not None:
                        self.tensorboard.add_figure('modes', fig_modes, global_step=step, close=True)
                    self.tensorboard.add_figure('latents', plot_latents(self.test_set, self.net), global_step=step, close=True)
                    self.tensorboard.add_scalar('test/loss', test_loss, step+1)
                    self.tensorboard.add_scalar('test/rel_Error', reco_error, step+1)
                    
                    
            if self.tensorboard:
                self.tensorboard.add_scalar('train_loss', loss, step+1)
                self.tensorboard.flush()
                
            lr_scheduler.step(loss)
            if step % 100 == 0:
                print(f"{step}: loss={loss.item():1.5}; test_loss={test_loss.item():1.5}", end="\r")
        return np.array(train_loss_log), np.array(test_loss_log)
    
    def get_reco_error(self, data):
        self.net.eval()
        with torch.no_grad():
            reco = self.net(data)[0]
        self.net.train()
        return torch.norm(data-reco, p='fro')/torch.norm(data, p='fro')
    
    def plot_test_idx_reco(self, plot_idx=None):
        plot_idx = np.random.randint(self.test_set.shape[0]) if plot_idx is None else plot_idx
        truth = to_torch(self.test_set[[plot_idx], ...], self.device)
        with torch.no_grad():
            self.net.eval()
            phi, q_hat = self.net(truth, return_phi=True)
            self.net.train()
        return plot_reconstrcution(truth, q_hat, phi)

    def plot_modes(self):
        fig = None
        if hasattr(self.net, 'decoder') and hasattr(self.net.decoder, 'get_modes'):
            modes = self.net.decoder.get_modes(detach=True, device='cpu')
            fig, axes = plt.subplots(1, modes.shape[0] , num=1, figsize=[19.2*modes.shape[0]/4, 10.8])
            axes = [axes] if modes.shape[0] == 1 else axes
            for n in range(modes.shape[0]):
                im = axes[n].imshow(modes[n, ...].T, origin='lower')
                plt.colorbar(im, ax=axes[n])
                axes[n].set_title('mode '+str(n))
        return fig

def plot_reconstrcution(truth, reco, phi=None):
    p1 = (phi is not None) * 1
    fig, axes = plt.subplots(1, 3 + p1, num=0, figsize=[19.2, 10.8])
    dims_remove = [0] * (truth.ndim - 2) + [...]
    if p1:
        pc = axes[0].imshow(phi[dims_remove].to('cpu').T, origin='lower')
        plt.colorbar(pc, ax=axes[0])
        axes[0].set_title('$ \phi $')
        axes[0].axis('equal')
    pc = axes[0 + p1].imshow(reco[dims_remove].to('cpu').T, origin='lower')
    axes[0 + p1].set_title('$ \^q $')
    axes[0 + p1].axis('equal')
    plt.colorbar(pc, ax=axes[0 + p1])
    pc = axes[1 + p1].imshow(truth[dims_remove].to('cpu').T, origin='lower')
    plt.colorbar(pc, ax=axes[1 + p1])
    axes[1 + p1].set_title('$ q $')
    axes[1 + p1].axis('equal')
    pc = axes[2 + p1].imshow((reco - truth)[dims_remove].abs().to('cpu').T, origin='lower')
    plt.colorbar(pc, ax=axes[2 + p1])
    axes[2 + p1].set_title('$| \^q - q |$')
    axes[2 + p1].axis('equal')
    return fig

def plot_latents(truth, net):
    net.eval()
    with torch.no_grad():
        code, phi, reco = net(truth, return_phi=True, return_code=True)
    net.train()
    fig = plt.figure()
    for n in range(code.shape[1]):
        plt.plot(code[:, n].to('cpu'), label=n)
    plt.legend()
    
    return fig
    
    

def show_video(truth, net, reps=10, pause=1):
    net.eval()
    with torch.no_grad():
        phi, reco = net(truth, return_phi=True)
    net.train()

    fig, axes = plt.subplots(1, 4, num=0, figsize=[19.2, 10.8])
    im = []
    for rep in range(reps):
        for n in range(truth.shape[0]):
            data = [phi[n, 0, ...].to('cpu'), reco[n, 0, ...].to('cpu'), truth[n, 0, ...].to('cpu'), (reco - truth)[n, 0, ...].abs().to('cpu')]
            titles = ['$ \phi $', '$ \^q $', '$ q $', '$| \^q - q |$']
            for m in range(4):
                if n == 0 and rep == 0:
                    im.append(axes[m].imshow(data[m].T, origin='lower'))
                    plt.colorbar(im[-1], ax=axes[m])
                    axes[m].set_title(titles[m])
                    axes[m].axis('equal')
                else:
                    im[m].set_data(data[m].T)
                    if m == 3:
                        im[m].set_clim(vmin=data[m].min(), vmax=data[m].max())
            plt.draw()
            plt.pause(pause)
    return fig