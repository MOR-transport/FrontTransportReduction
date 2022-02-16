#######################################################################################################################
#           GLOBALS for the examples folder
#######################################################################################################################

# import the lib folder in which all sources are kept:
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../lib")
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using: "+ torch.cuda.get_device_name(0))

# plotting
import matplotlib

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
latexfont = {'size'   : 28}
matplotlib.rc('font',**latexfont)
matplotlib.rc('text',usetex=True)
matplotlib.use('TkAgg')

