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
    
    
class ConvTConv(BaseModule):
    def __init__(self, kernel_size=[5, 11], channels=[1, 16, 1]):
        super(self.__class__, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size[0],
                              padding=int((kernel_size[0] - 1)/2), padding_mode='replicate')
        self.bn = nn.BatchNorm2d(channels[1])
        self.pad = nn.ReplicationPad2d(int((kernel_size[1]-1)/2))
        self.unpad_size = kernel_size[1] - 1
        self.Tconv = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[-1], kernel_size=kernel_size[1])

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        out = self.Tconv(self.pad(out))[..., self.unpad_size:-self.unpad_size, self.unpad_size:-self.unpad_size]
    
        return out

# class ODE_Block(BaseModule):
#     def __init__(self,)

class Phi2(BaseModule):
    def __init__(self, n_layers=16, f=nn.Sigmoid):
        super(self.__class__, self).__init__()
        self.has_bottleneck = False
        self.f = f()
        self.instnorm = nn.InstanceNorm2d(1)
        #layer_act_bn = nn.Sequential(ConvTConv(), nn.BatchNorm2d(1), self.act)
        self.layers = nn.ModuleList([nn.Sequential(ConvTConv(), nn.BatchNorm2d(1), self.act) for _ in range(n_layers-1)])
        self.layers.append(ConvTConv())
        
    def forward(self, q, apply_f=True, return_phi=True):
        phi = self.instnorm(q)
        for layer in self.layers:
            phi = layer(phi)
        if apply_f and return_phi:
            return phi, self.f(phi)
        if apply_f:
            return self.f(phi)
        return phi
    
    
class ResLayer(BaseModule):
    def __init__(self, kernel_size=3, channels=16, T=False):
        super(self.__class__, self).__init__()
        channels = [channels] if isinstance(channels, int) else channels
        #self.pad1 = nn.ReplicationPad2d(int((kernel_size-1)/2))
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[-1], kernel_size=kernel_size,
                                            padding=int((kernel_size - 1)/2), padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(channels[-1])
        if T:
            self.conv2 = nn.ConvTranspose2d(in_channels=channels[-1], out_channels=channels[0], kernel_size=kernel_size,
                                            padding=int((kernel_size - 1)/2))
        else:
            self.conv2 = nn.Conv2d(in_channels=channels[-1], out_channels=channels[0], kernel_size=kernel_size,
                                    padding=int((kernel_size - 1)/2), padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(channels[0])
        
    def forward(self, x):
        
        out = self.bn1(self.act(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return x + out
    
    
class ResBlock(BaseModule):
    def __init__(self, kernel_size=3, channels=16, layers=5, act_output=True):
        super(self.__class__, self).__init__()
        channels = [channels] * 2 if isinstance(channels, int) else channels
        seq_layers = [nn.Sequential(ResLayer(kernel_size, channels), self.act, nn.BatchNorm2d(channels[0]))
                      for _ in range(layers - act_output)]
        if not act_output:
            seq_layers.append(ResLayer(kernel_size, channels))
            
        self.block = nn.Sequential(*seq_layers)
    
    def forward(self, x):
        return self.block(x)
        
        
class Phi(BaseModule):
    def __init__(self, kernel_size=9, channels=[8], blocks=4, layers_per_block=3, f=nn.Sigmoid):
        super(self.__class__, self).__init__()
        self.has_bottleneck = False
        self.f = f()
        self.instnorm = nn.InstanceNorm2d(1)
        self.pad = nn.ReplicationPad2d(int((kernel_size-1)/2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[-1], kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(channels[-1])
        self.resblocks = nn.Sequential(*[ResBlock(kernel_size, channels, layers_per_block) for _ in range(blocks)])
        self.conv_out = nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=1)
        
    def forward(self, q, apply_f=True, return_phi=True):
        q = self.bn1(self.act(self.pad(self.conv1(self.instnorm(q)))))
        phi = self.conv_out(self.resblocks(q))
        #phi = self.resblocks(q)
        if apply_f and return_phi:
            return phi, self.f(phi)
        if apply_f:
            return self.f(phi)
        return phi
    
    
class FTR_Enc_small(BaseModule):
    def __init__(self, n_alphas=10, output_activation=False):
        super(self.__class__, self).__init__()
        self.instnorm = nn.InstanceNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.out_fc1 = nn.Linear(2704, n_alphas)


class FTR_Enc(BaseModule):
    def __init__(self, n_alphas=10, output_activation=False, spatial_shape=[128, 128], input_norm=nn.InstanceNorm2d):
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
        self.out_fc2 = nn.Linear(512, n_alphas)

        # self.out_act = nn.ReLU()
        if output_activation:
            self.out_act = self.act
        else:
            self.out_act = None
    
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
        out = self.out_act(out) + isinstance(self.act, nn.ELU) if self.out_act is not None else out
        return out


class FTR_Enc2(BaseModule):
    def __init__(self, n_alphas=10, output_activation=False, spatial_shape=[128, 128], input_norm=nn.BatchNorm2d):
        super(self.__class__, self).__init__()
        k = 5
        s = 2
        self.spatial_shape = spatial_shape
        self.input_norm = input_norm(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=k)
        self.bn1 = nn.BatchNorm2d(16)
        self.res_block1 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.spatial_shape = [int((n - k)) + 1 for n in self.spatial_shape]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.bn2 = nn.BatchNorm2d(16)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        self.res_block2 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.bn3 = nn.BatchNorm2d(16)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        self.res_block3 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.bn4 = nn.BatchNorm2d(16)
        self.spatial_shape = [int((n - k) / s) + 1 for n in self.spatial_shape]
        self.res_block4 = ResBlock(kernel_size=5, channels=16, layers=2)

        self.out_fc0 = nn.Linear(16 * self.spatial_shape[0] * self.spatial_shape[1], 2048)
        self.bn_out_fc0 = nn.BatchNorm1d(2048)
        self.out_fc1 = nn.Linear(2048, 512)
        self.bn_out_fc1 = nn.BatchNorm1d(512)
        self.out_fc2 = nn.Linear(512, n_alphas)
        
        # self.out_act = nn.ReLU()
        if output_activation:
            self.out_act = self.act
        else:
            self.out_act = None
    
    def forward(self, q):
        q = self.input_norm(q)
        
        # feature generation:
        q = self.res_block1(self.bn1(self.act(self.conv1(q))))
        q = self.res_block2(self.bn2(self.act(self.conv2(q))))
        q = self.res_block3(self.bn3(self.act(self.conv3(q))))
        q = self.res_block4(self.bn4(self.act(self.conv4(q))))
        
        out = q.flatten(start_dim=1)
        out = self.bn_out_fc0(self.act(self.out_fc0(out)))
        out = self.bn_out_fc1(self.act(self.out_fc1(out)))
        out = self.out_fc2(out)
        out = self.out_act(out) + isinstance(self.act, nn.ELU) if self.out_act is not None else out
        return out
    
    
class FTR_Dec(BaseModule):
    def __init__(self, n_alpha=10, f=nn.Sigmoid, learn_frontwidth=False, spatial_shape=[128, 128]):
        super(self.__class__, self).__init__()
        
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
            
        self.learn_frontwidth = learn_frontwidth
        self.pad = nn.ReplicationPad2d(2)
        self.in_fc1 = nn.Linear(n_alpha - learn_frontwidth, 512)
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
        lam = 1
        if self.learn_frontwidth:
            lam = alpha[:, [0]][..., None, None].abs()  # shape: batch, x, y
            alpha = alpha[:, 1:]
        
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
                return phi, self.f(phi * lam)
            else:
                return self.f(phi * lam)
        else:
            return phi


class FTR_Dec2(BaseModule):
    def __init__(self, n_alpha=10, f=nn.Sigmoid, learn_frontwidth=False, spatial_shape=[128, 128]):
        super(self.__class__, self).__init__()
        
        k = 5
        s = 2
        shape0 = [spatial_shape]
        for lay in range(4):
            shape0.append([int((n - k) / (s if lay > 0 else 1)) + 1 for n in shape0[-1]])
        self.spatial_shape0 = shape0[::-1]
        self.unpad = []
        for n in range(len(self.spatial_shape0) - 1):
            s0 = (1 if n == (len(self.spatial_shape0) - 2) else s)
            self.unpad.append([m - ((l - 1) * s0 + k) for l, m in zip(self.spatial_shape0[n], self.spatial_shape0[n + 1])])
        
        self.learn_frontwidth = learn_frontwidth
        self.pad = nn.ReplicationPad2d(2)
        
        self.in_fc0 = nn.Linear(n_alpha - learn_frontwidth, 512)
        self.bn_in_fc0 = nn.BatchNorm1d(512)
        self.in_fc1 = nn.Linear(512, 2048)
        self.bn_in_fc1 = nn.BatchNorm1d(2048)
        self.in_fc2 = nn.Linear(2048, 16 * self.spatial_shape0[0][0] * self.spatial_shape0[0][1])
        
        
        self.bn1 = nn.BatchNorm2d(16)
        self.res_block1 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.tconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.bn2 = nn.BatchNorm2d(16)
        self.res_block2 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.tconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.bn3 = nn.BatchNorm2d(16)
        self.res_block3 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.tconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=k, stride=s)
        self.res_block4 = ResBlock(kernel_size=5, channels=16, layers=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.tconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=k)
        self.res_block5 = ResBlock(kernel_size=5, channels=1, layers=2, act_output=False)
        
        self.f = f()
    
    def forward(self, alpha, apply_f=True, return_phi=False):
        lam = 1
        if self.learn_frontwidth:
            lam = alpha[:, [0]][..., None, None].abs()  # shape: batch, x, y
            alpha = alpha[:, 1:]
        
        alpha = self.bn_in_fc0(self.act(self.in_fc0(alpha)))
        alpha = self.bn_in_fc1(self.act(self.in_fc1(alpha)))
        alpha = self.act(self.in_fc2(alpha))
        
        phi = self.bn1(alpha.reshape(-1, 16, *self.spatial_shape0[0]))
        
        # feature generation:
        phi = self.bn2(self.act(self.tconv1(self.pad(self.res_block1(phi)))))[..., 4:-(4 - self.unpad[0][0]), 4:-(4 - self.unpad[0][1])]
        phi = self.bn3(self.act(self.tconv2(self.pad(self.res_block2(phi)))))[..., 4:-(4 - self.unpad[1][0]), 4:-(4 - self.unpad[1][1])]
        phi = self.bn4(self.act(self.tconv3(self.pad(self.res_block3(phi)))))[..., 4:-(4 - self.unpad[2][0]), 4:-(4 - self.unpad[2][1])]
        phi = self.res_block5(self.tconv4(self.res_block4(phi)))
        
        if apply_f:
            if return_phi:
                return phi, self.f(phi * lam)
            else:
                return self.f(phi * lam)
        else:
            return phi


class FTR_Dec_1Lay(BaseModule):
    def __init__(self, n_alpha=10, f=nn.Sigmoid, learn_frontwidth=False, spatial_shape=[128, 128]):
        super(self.__class__, self).__init__()
        self.spatial_shape = spatial_shape
        self.in_fc1 = nn.Linear(n_alpha - learn_frontwidth, self.spatial_shape[0] * self.spatial_shape[1], bias=False)
        self.f = f()
        self.learn_frontwidth = learn_frontwidth
    
    def forward(self, alpha, apply_f=True, return_phi=False):
        
        if self.learn_frontwidth:
            lam = alpha[:, [0]][..., None, None].abs()  # shape: batch, x, y
            phi = self.in_fc1(alpha[:, 1:]).reshape(alpha.shape[0], 1, self.spatial_shape[0], self.spatial_shape[1])
        else:
            lam = 1
            phi = self.in_fc1(alpha).reshape(alpha.shape[0], 1, self.spatial_shape[0], self.spatial_shape[1])
        
        if apply_f:
            if return_phi:
                return phi, self.f(phi * lam)
            else:
                return self.f(phi * lam)
        else:
            return phi
    
    def get_modes(self, detach=False, device=DEVICE):
        modes = self.in_fc1.weight.reshape(*self.spatial_shape, -1).permute(-1, 0, 1)
        if detach:
            modes = modes.detach().to(device)
        return modes


class FTR_AE(BaseModule):
    def __init__(self, encoder=None, decoder=None, n_alphas=10, f=nn.Sigmoid, learn_periodic_alphas=True, alpha_act=True, spatial_shape=[128, 128]):
        super(self.__class__, self).__init__()
        self.has_bottleneck = True
        self.encoder = FTR_Enc(n_alphas, alpha_act, spatial_shape=spatial_shape) if encoder is None else encoder
        self.decoder = FTR_Dec(n_alphas, f) if decoder is None else decoder

        self.periodic_net = PeriodicFunction(n_alpha=n_alphas, n_freqs=n_alphas) if learn_periodic_alphas else None

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


class PeriodicFunction(BaseModule):
    def __init__(self, n_alpha=10, n_freqs=10):
        super(self.__class__, self).__init__()
        self.omega = nn.Linear(1, n_freqs, bias=False)
        self.alpha = nn.Linear(2 * n_freqs, n_alpha)
    
    def forward(self, t):
        om_t = self.omega(t)
        return self.alpha(torch.cat([torch.cos(om_t), torch.sin(om_t)], -1))