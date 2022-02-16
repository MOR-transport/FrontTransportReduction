function dudt = jac_rhs_react(t,u,DU,kappa,delta)
% rhs of the PDE: dudt = ddudxx + 8/delta^2*u(1-u)
% with exact solution of the form
% u(x,t) = 1/2(1-tanh((x-2t/delta)/delta)
DUhat = fft(DU);
ddDUhat = -kappa.^2.*DUhat;
ddDU = real(ifft(ddDUhat));
dudt = ddDU + 8/delta.^2 * (2-3*u).*u.*DU;