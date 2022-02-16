
import numpy as np
from FOM.FOM import params_class
from ROM.ROM import rom_params_class


class my_cases(params_class):
    def __init__(self, case="bunsen_flame", reaction_const=None):
        if case == "bunsen_flame":
            super(my_cases, self).__init__(pde="react", dim = 2, L=[1, 0.5], N = [2**8,2**7], T=20,  Nt=300, case="bunsen_flame")
            self.advection_speed = 0.05  # 0.1 # velocity of moving vortex
            self.w0 = [5, 5]  # initial vorticity
            self.r0 = 0.002  # initial size of vortex
            self.decay_constant = 0.2  # decay of the initial vorticity
            # init advection reaction diffusion
            self.diffusion_const = 0.5e-4  # adjust the front width ( larger value = larger front)
            self.reaction_const = 1  # adjust the propagation of the flame (larger value = faster propagation)
            self.penalisation.eta = 1e2
            self.penalisation.init_mask(self)
            self.inicond = self.set_inicond(case=case)
            self.rom = rom_params_class(self, rom_size=10, fom_size=self.fom_size, time_points= self.time.t, online_prediction_method = "lspg-hyper", mapping_type = "FTR")
            self.rom.ftr.max_iter = 300
            self.rom.fom.sampleMeshSize = 300
        elif case == "pacman":
            super(my_cases, self).__init__(pde="react", dim=2, L=[1, 1], N = [2**9]*2, T=3,  Nt=200 ,case="pacman")
            self.advection_speed= 0.1#0.1 # velocity of moving vortex
            self.w0 = [30, 30]   # initial vorticity
            self.r0 = 0.005  # initial size of vortex
            self.decay_constant = 0.2 # decay of the initial vorticity
            # init advection reaction diffusion
            self.diffusion_const = 1e-4  # adjust the front width ( larger value = larger front)
            self.reaction_const = 10 # adjust the propagation of the flame (larger value = faster propagation)
            self.penalisation.case = case
            self.penalisation.init_mask(self)
            self.inicond = self.set_inicond(case=case)
            self.odeint.timestep = self.time.dt/4
            self.rom = rom_params_class(self, rom_size=10, fom_size=self.fom_size, time_points=self.time.t,
                                        online_prediction_method="galerkin-hyper", mapping_type="FTR")
            self.rom.fom.sampleMeshSize = 2**(16)//10
            self.rom.ftr.max_iter = 1000
            self.odeint.timestep = self.time.dt/4
            self.rom.ftr.offset = -1
        elif case == "pacmanPOD":
            super(my_cases, self).__init__(pde="react", dim=2, L=[1, 1], N=[2 ** 9] * 2, T=3, Nt=200, case="pacman")
            self.advection_speed = 0.1  # 0.1 # velocity of moving vortex
            self.w0 = [30, 30]  # initial vorticity
            self.r0 = 0.005  # initial size of vortex
            self.decay_constant = 0.2  # decay of the initial vorticity
            # init advection reaction diffusion
            self.diffusion_const = 1e-4  # adjust the front width ( larger value = larger front)
            self.reaction_const = 10  # adjust the propagation of the flame (larger value = faster propagation)
            self.penalisation.case = case
            self.penalisation.init_mask(self)
            self.inicond = self.set_inicond(case=case)
            self.odeint.timestep = self.time.dt / 4
            self.rom = rom_params_class(self, rom_size=10, fom_size=self.fom_size, time_points=self.time.t,
                                        online_prediction_method="galerkin", mapping_type="POD")
            self.rom.fom.sampleMeshSize = 2 ** (16) // 10
            self.rom.ftr.max_iter = 1000
            self.odeint.timestep = self.time.dt / 4
        elif case == "reaction1D":
            super(my_cases, self).__init__(pde="react", dim=1, L=[30], N=[4000], T=1, Nt=100, case="reaction1D")
            self.reaction_const = 0.2#reaction_const  # adjust the propagation of the flame (larger value = faster propagation)
            self.inicond = self.set_inicond(case=case)
            self.rom = rom_params_class(self, rom_size=4, fom_size=self.fom_size, time_points=self.time.t,
                                        online_prediction_method="galerkin-hyper", mapping_type="FTR")
            self.odeint.timestep = self.time.dt /400
            self.rom.fom.sampleMeshSize = 400
            self.rom.ftr.max_iter = 5000
            self.rom.ftr.offset = -1

        elif case == "disc":
            super(my_cases, self).__init__(pde="advection", dim=2, L=[1, 1], N=[2 ** 7] * 2, T=3, Nt=100, case="pacman")

