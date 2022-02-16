import os
import re
import torch
from shutil import copy2 as copy_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch(data, device=DEVICE):
    if (type(data) == list) or (type(data) == tuple):
        data = [to_torch(el, device) for el in data]
    elif type(data) == dict:
        data = {key: to_torch(data[key], device) for key in data}
    else:
        if type(data) != torch.Tensor:
            data = torch.tensor(data, device=device, dtype=torch.float32)
        else:
            data = data.to(device)
    return data


def to_numpy(data):
    if (type(data) == list) or (type(data) == tuple):
        data = [to_numpy(el) for el in data]
    elif type(data) == dict:
        data = {key: to_numpy(data[key]) for key in data}
    else:
        if type(data) == torch.Tensor:
            data = data.data.to('cpu').numpy()
    return data

def frobenius2(Tensor):
    reduce_dims = [n + 1 for n in range(Tensor.ndim - 1)]
    return Tensor.square().sum(reduce_dims)

def nuclear(Tensors):
    snapshot_matrix = Tensors.flatten(1)
    _, s, _ = snapshot_matrix.svd()
    return s.sum()

def save_codeBase(source_path, dest_path):
    source_path = source_path if source_path.endswith('/') else source_path + '/'
    dest_path = dest_path + 'code/' if dest_path.endswith('/') else dest_path + '/code/'
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    for fname in os.listdir(source_path):
        if fname.endswith('.py'):
            copy_file(source_path + fname, dest_path + fname)
            
# def plot_modes(decoder):
#     modes = decoder.in_fc1.weights.detach().to('cpu').numpy()
#     for mode in modes.shape[]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

