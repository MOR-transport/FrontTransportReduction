import torch as to
import torch.nn as nn
from torch import optim
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

if to.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# dev="cpu"
device = to.device("cpu")

def build_ECSW_system( mapping , rom_states, fom_states, parameter_list):
    import numpy as np
    from time import perf_counter

    N_params = len(parameter_list)
    time_list = parameter_list[0].time.t
    N_time = len(parameter_list[0].time.t)

    U = mapping.U
    M,rank = np.shape(U)

    G = np.zeros([N_time*N_params*rank,M])
    b = np.zeros([N_time*N_params*rank])

    n  = 0
    ns = 0
    for k, params in enumerate(parameter_list):
        for it, time in enumerate(time_list):
            tcpu = perf_counter()
            rhs = params.rhs

            q = fom_states[:,ns]
            a = rom_states[:,ns]
            q_tilde = mapping.applyMapping(a)

            mapping.updateJacobian( a )
            J = mapping.jacobian()

            q_dot = rhs(q, time)
            a_dot =  np.linalg.lstsq(J, q_dot)[0]

            R = J@a_dot - rhs(q_tilde, time)

            for ix in range(M):
                Re = np.take(R, ix, axis = 0)
                Je = np.take(J, ix, axis = 0)
                G[n:n+rank, ix] = Je.T * Re

            b[n:n+rank] = J.T @ R
            n += rank
            ns+= 1

            print(f"it: {ns}, t_cpu={perf_counter()-tcpu}")

    # sanity check: G@1 = b
    one = np.ones([M,1])
    err = np.linalg.norm(np.squeeze(G@one)-b)/np.linalg.norm(b)
    print( f"Sanity check! Relative error || G*1-b||/||b|| = {err}\n")

    return G, b



def generate_train_and_test_data(G, b, rom_states, mu_vals, idx_train, it_start):
    Nsamples = len(mu_vals)
    Nmodes, Ntime = np.shape(rom_states)
    Ntime = Ntime//Nsamples
    Ne = np.size(G,1)
    Gs = to.tensor(G, dtype=to.float32).reshape(Nsamples, Ntime, Nmodes, Ne)
    bs = (Gs.clone().detach() @ to.ones(Ne)).reshape(Nsamples, Ntime, Nmodes)
    xs = to.tensor(rom_states, dtype=to.float32).moveaxis(-1, 0).reshape(Nsamples, Ntime, Nmodes)

    mu_vals = to.tensor(mu_vals, dtype=to.float32)
    mudat = to.cat([mu_vals[i] * to.ones(Ntime) for i in range(len(mu_vals))]).reshape(Nsamples, Ntime)


    # split train test
    G_data = {"train": [], "test": []}
    b_data = {"train": [], "test": []}
    x_data = {"train": [], "test": []}
    mu_data = {"train": [], "test": []}
    for idx in range(Nsamples):
        G = Gs[idx, ...]
        b = bs[idx, ...]
        x = xs[idx, ...]
        mu = mudat[idx, ...]
        if idx in idx_train:
            G_data["train"].append(G[it_start:])
            b_data["train"].append(b[it_start:])
            x_data["train"].append(x[it_start:])
            mu_data["train"].append(mu[it_start:])
        else:
            G_data["test"].append(G[it_start:])
            b_data["test"].append(b[it_start:])
            x_data["test"].append(x[it_start:])
            mu_data["test"].append(mu[it_start:])

    G_data["train"] = to.cat(G_data["train"])
    b_data["train"] = to.cat(b_data["train"])
    x_data["train"] = to.cat(x_data["train"])
    mu_data["train"] = to.cat(mu_data["train"]).reshape([-1, 1])
    train_set = to.utils.data.TensorDataset(G_data["train"], b_data["train"], mu_data["train"], x_data["train"])


    G_data["test"] = to.cat(G_data["test"])
    b_data["test"] = to.cat(b_data["test"])
    x_data["test"] = to.cat(x_data["test"])
    mu_data["test"] = to.cat(mu_data["test"]).reshape([-1, 1])
    test_set = to.utils.data.TensorDataset(G_data["test"], b_data["test"], mu_data["test"], x_data["test"])

    return train_set, test_set

def FTR_hyper_loss(alpha, G, b, lambd =3e-6):


    l1=to.sum(to.abs(alpha))
    l2loss = to.nn.MSELoss(reduction="sum")
    l2=0
    for i in range(G.size(0)):
        l2 +=l2loss(to.matmul(G[i], alpha[i].T),b[i].T)
        #l2 += l2loss(alpha[i].T, to.matmul(to.pinverse(G[i]), b[i].T))
    loss = l1*lambd + l2

    return loss,l2

class NeuralNetwork1(nn.Module):
    def __init__(self, Ur):
        super(NeuralNetwork1, self).__init__()
        self.flatten = nn.Flatten()
        self.phi_modes = to.tensor(Ur,requires_grad=False).to(device)
        self.size_modes= np.size(Ur,0)
        self.front =lambda x: 0.5*(1-to.tanh(x/2))
        self.dfront = lambda x: - 0.25/ (to.cosh(x / 2)** 2)
        self.mu_map= nn.Sequential(
            nn.Linear(1,10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid()
        )
        self.linear_relu_stack = nn.Sequential(
             nn.Linear(self.size_modes+10, self.size_modes//4, bias=True),
             nn.Dropout(p=0.5),
             nn.Sigmoid(),
            nn.Linear(self.size_modes//4, self.size_modes, bias=False),
            #nn.Dropout(p=0.5),
            nn.ReLU()
        )

    def forward(self, x, mu):
        x = self.phi_modes @ x.T
        y = self.mu_map(mu)
        x = self.front(x).T
        #y=mu
        z = to.cat([x,y],dim=1)
        logits = self.linear_relu_stack(z)
        return logits

class NeuralNetwork2(nn.Module):
    def __init__(self, Ur):
        super(NeuralNetwork2, self).__init__()
        self.flatten = nn.Flatten()
        self.phi_modes = to.tensor(Ur,requires_grad=False).to(device)
        self.size_modes = np.size(Ur, 0)
        self.front =lambda x: 0.5*(1-to.tanh(x/2))
        self.dfront = lambda x: - 0.25/ (to.cosh(x / 2)** 2)
        self.hard = nn.Hardsigmoid()
        self.mu_map= nn.Sequential(
            nn.Linear(1,10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid()
        )
        self.linear_relu_stack = nn.Sequential(
             nn.Linear(5, self.size_modes//4, bias=True),
             nn.Sigmoid(),
             nn.Linear(self.size_modes//4, self.size_modes//4, bias=False),
             nn.ReLU()
        )

    def forward(self, x, mu):
        # x = self.phi_modes @ x.T
        # x = self.front(x).T
        # y = self.mu_map(mu)
        z = to.cat([x,mu],dim=1)
        logits = self.linear_relu_stack(z)
        return logits



def init_weights(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        #m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0)



# %%

def solve_NN_EECSW(G, b, rom_states, Ur, mu, idx_train, maxiter = 22000, batch_size=100, it_start = 0, learning_rate = 1e-5):



    train_set,  test_set = generate_train_and_test_data(G, b, rom_states, mu, idx_train, it_start = it_start )

    train_loader = to.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                            collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = to.utils.data.DataLoader(test_set, batch_size=batch_size,
                                           collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    Ur = to.tensor(Ur, dtype=to.float32)
    network = NeuralNetwork1(Ur)
    network.apply(init_weights)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    for t in range(maxiter):
        # Forward pass: compute predicted y by passing x to the model.
        losses = []
        mean_loss = 0

        for batch_ndx, (G, b, mu, x) in enumerate(train_loader):

            y_pred = network(x,mu)

            # Compute and print loss.
            loss,loss2 = FTR_hyper_loss(y_pred, G, b)
            loss2 /= to.sum(b ** 2)
            losses.append(loss2)
            mean_loss += loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        nloss = len(losses)
        if np.mod(t,100)==0:
            print("epoch train: %d  %1.1e > loss > %1.1e (mean: %1.1e), loss: %1.1e, len %d"%(t,max(losses).detach().cpu().numpy(),min(losses).detach().cpu().numpy(), mean_loss/nloss, loss, max(sum(y_pred.T>0))))
        losses = []
        mean_loss = 0
        #if
        with to.set_grad_enabled(False):
            for batch_ndx, (G, b, mu, x) in enumerate(test_loader):
                y_pred = network(x, mu)

                # Compute and print loss.
                loss,loss2 = FTR_hyper_loss(y_pred, G, b)
                loss2 /= to.sum(b ** 2)
                losses.append(loss2.cpu().detach().numpy())

            if np.mod(t, 100) == 0:
                print("epoch test: %d  %1.1e > loss > %1.1e (mean: %1.1e), loss: %1.1e, len %d" % (
                t, max(losses), min(losses),np.mean(losses), loss,
                max(sum(y_pred.T > 0))))

    return network, test_set