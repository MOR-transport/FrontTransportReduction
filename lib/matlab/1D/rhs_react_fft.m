function dudt = rhs_react(t,u,kappa,delta)
% rhs of the PDE: dudt = ddudxx + 8/delta^2*u(1-u)
% with exact solution of the form
% u(x,t) = 1/2(1-tanh((x-2t/delta)/delta)
uhat = fft(u);
dduhat = -kappa.^2.*uhat;
ddu = real(ifft(dduhat));
dudt = ddu + 8/delta.^2 * u.^2.*(1-u);