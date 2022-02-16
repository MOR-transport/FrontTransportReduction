import torch.nn as nn
import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModule(nn.Module):
    def __init__(self, activation=nn.ELU):
        super(BaseModule, self).__init__()
        self.act = activation()

    def save_net_weights(self, fpath, fname):
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        torch.save(self.state_dict(), fpath + fname)

    def load_net_weights(self, fpath):
        self.load_state_dict(torch.load(fpath, map_location=DEVICE))
    
    
class FTR_Enc(BaseModule):
    def __init__(self, dof=10, spatial_shape=[128, 128], input_norm=nn.InstanceNorm2d):
        super(self.__class__, self).__init__()
        k = 5
        s = 2
        self.spatial_shape = spatial_shape
        self.input_norm = input_norm(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=k)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial_shape = [int((n - k)) + 1 for n in self.spatial_shape]
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=k, stride=s)
        self.bn2 = nn.BatchNorm2d(16)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=k, stride=s)
        self.bn3 = nn.BatchNorm2d(32)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=k, stride=s)
        self.bn4 = nn.BatchNorm2d(16)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        
        self.out_fc1 = nn.Linear(16 * self.spatial_shape[0] * self.spatial_shape[1], 512)
        self.bn_out_fc1 = nn.BatchNorm1d(512)
        self.out_fc2 = nn.Linear(512, dof)

    
    def forward(self, q):
        q = self.input_norm(q)
        
        # feature generation:
        q = self.bn1(self.act(self.conv1(q)))
        q = self.bn2(self.act(self.conv2(q)))
        q = self.bn3(self.act(self.conv3(q)))
        q = self.bn4(self.act(self.conv4(q)))
        
        out = q.flatten(start_dim=1)
        out = self.bn_out_fc1(self.act(self.out_fc1(out)))
        out = self.out_fc2(out)
        return out
    
    
class FTR_Dec(BaseModule):
    def __init__(self, dof=10, f=nn.Sigmoid, spatial_shape=[128, 128], lam=1, init_zero=False):
        super(self.__class__, self).__init__()

        self.lam = lam
        
        k = 5
        s = 2
        shape0 = [spatial_shape]
        for lay in range(4):
            shape0.append([int((n - k) / (s if lay > 0 else 1)) + 1 for n in shape0[-1]])
        self.spatial_shape0 = shape0[::-1]
        self.unpad = []
        for n in range(len(self.spatial_shape0) - 1):
            s0 = (1 if n == (len(self.spatial_shape0) - 2) else s)
            self.unpad.append([m - ((l - 1) * s0  + k) for l, m in zip(self.spatial_shape0[n], self.spatial_shape0[n+1])])

        self.pad = nn.ReplicationPad2d(2)
        self.in_fc1 = nn.Linear(dof, 512)
        self.bn_in_fc1 = nn.BatchNorm1d(512)
        self.in_fc2 = nn.Linear(512, 16 * self.spatial_shape0[0][0] * self.spatial_shape0[0][1])
        
        self.bn1 = nn.BatchNorm2d(16)
        self.tconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=k, stride=s)
        self.bn2 = nn.BatchNorm2d(32)
        self.tconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=k, stride=s)
        self.bn3 = nn.BatchNorm2d(16)
        self.tconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=k, stride=s)
        self.bn4 = nn.BatchNorm2d(8)
        self.tconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=k)
        self.f = f()
    
    def forward(self, alpha, apply_f=True, return_phi=False):
        
        alpha = self.bn_in_fc1(self.act(self.in_fc1(alpha)))
        alpha = self.act(self.in_fc2(alpha))
        
        phi = self.bn1(alpha.reshape(-1, 16, *self.spatial_shape0[0]))
        
        # feature generation:
        phi = self.bn2(self.act(self.tconv1(self.pad(phi))))[..., 4:-(4 - self.unpad[0][0]), 4:-(4 - self.unpad[0][1])]
        phi = self.bn3(self.act(self.tconv2(self.pad(phi))))[..., 4:-(4 - self.unpad[1][0]), 4:-(4 - self.unpad[1][1])]
        phi = self.bn4(self.act(self.tconv3(self.pad(phi))))[..., 4:-(4 - self.unpad[2][0]), 4:-(4 - self.unpad[2][1])]
        phi = self.tconv4(phi)
        
        if apply_f:
            if return_phi:
                return phi, self.f(phi/self.lam)
            else:
                return self.f(phi/self.lam)
        else:
            return phi


class FTR_Dec_1Lay(BaseModule):
    def __init__(self, dof=10, f=nn.Sigmoid, spatial_shape=[128, 128], lam=1, init_zero=True):
        super(self.__class__, self).__init__()
        self.spatial_shape = spatial_shape
        self.lam = lam
        self.in_fc1 = nn.Linear(dof, self.spatial_shape[0] * self.spatial_shape[1], bias=False)
        if init_zero:
            nn.init.constant_(self.in_fc1.weight, 0)
        self.f = f()
    
    def forward(self, alpha, apply_f=True, return_phi=False):
        
        phi = self.in_fc1(alpha).reshape(alpha.shape[0], 1, self.spatial_shape[0], self.spatial_shape[1])
        
        if apply_f:
            if return_phi:
                return phi, self.f(phi/self.lam)
            else:
                return self.f(phi/self.lam)
        else:
            return phi
    
    def get_modes(self, detach=False, device=DEVICE):
        modes = self.in_fc1.weight.reshape(*self.spatial_shape, -1).permute(-1, 0, 1)
        if detach:
            modes = modes.detach().to(device)
        return modes


class FTR_AE(BaseModule):
    def __init__(self, encoder=None, decoder=None, dof=10, f=nn.Sigmoid, spatial_shape=[128, 128]):
        super(self.__class__, self).__init__()
        self.has_bottleneck = True
        self.encoder = FTR_Enc(dof, spatial_shape=spatial_shape) if encoder is None else encoder
        self.decoder = FTR_Dec(dof, f) if decoder is None else decoder

    def forward(self, q, return_code=False, apply_f=True, return_phi=False):
        code = self.encoder(q)
        returns = []
        if return_code:
            returns.append(code)
        if apply_f and return_phi:
            returns += [*self.decoder(code, apply_f=apply_f, return_phi=return_phi)]
        else:
            returns += [self.decoder(code, apply_f)]
            
        return returns
