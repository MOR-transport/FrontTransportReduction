clear all, close all, clc
addpath(genpath("lib"))

% Define spatial domain
c_train = [-2,2];              % Wave speed
c_test = [0.5];              % Wave speed
L = 20;             % Length of domain
N = 1000;           % Number of discretization points
dx = L/N;
x = -L/2:dx:L/2-dx; % Define x domain


front = @(x) 0.5*(1-tanh(x*5/2));
dfront = @(x) -2.5./(cosh(x*5/2).^2);
%dfront = @(x) front(x).*(1-front(x))*10;
% front = @(x) 1./(1+exp(-x*10));
% dfront = @(x) front(x).*(1-front(x))*10;
% Define discrete wavenumbers
kappa = (2*pi/L)*[-N/2:N/2-1];
kappa = fftshift(kappa');    % Re-order fft wavenumbers

h = x(2)-x(1);
koefDX = [-1/60 	3/20 	-3/4 	0 	3/4 	-3/20 	1/60] ;
%koefDX = [1/280, 	-4/105 ,	1/5, 	-4/5, 	0 ,	4/5, 	-1/5, 	4/105 ,	-1/280 ] ;
%koefDXX=  [ 0 1 -2 1  0 ] ;
koefDXX= [-1/12 	4/3 	-5/2 	4/3 	-1/12] ;
%koefDXX = [-1/560 	8/315 	-1/5 	8/5 	-205/72 	8/5 	-1/5 	8/315 	-1/560];
Dx  = D_generalPeriodic(N, h  ,koefDX  ) ;

% Initial condition
u0 = front(abs(x)-2);

% Simulate in Fourier frequency domain
dt = 0.025;
t = 0:dt:100*dt;

[X,T] = meshgrid(x,t);

phi = zeros(N,length(t),length(c_train));
q = zeros(N,length(t),length(c_train));
i=0;
for c = c_train
    i = i +1;
    phi_temp =sin(2*pi*(X-c*T)/L);
    phi(:,:,i) = phi_temp';
    %q(:,:,i) = front(phi_temp');
    % Alternatively, simulate in spatial domain
    [t,u] = ode45(@(t,u)rhsWaveSpatial(t,u,kappa,c),t,u0);
    q(:,:,i) = u';
end
u0 = q(:,1,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r=5;
if 0
    [U,S,V] = svds(reshape(phi,N,[]),r);
    Ur = U;
    mapping = @(a) front(Ur*a);
    jac =  @(a) diag(dfront(Ur*a))*Ur;
else
    phi = FTR_direct(reshape(q,N,[]), front, r, 3000);
    phi_mat = reshape(phi,N,[]);
    [U,S,V] = svds(phi_mat,r);
    Ur = U;
    ampl = S*V';
    mapping = @(a) front(Ur*a);
    %jac =  @(a) diag(dfront(Ur*a))*Ur;
    jac = @(a)  diff_mapping(mapping,a,r);

end



%% ROM test parameter
a = S*V';
advect = @(time) 5* sin(2*pi * time/t(end));
a0 = a(:,1);
rhs = @(t,a) jac(a)\rhsWaveSpatial(t,mapping(a),kappa,advect(t));
[t,a] = ode45(rhs,t,a0);
qtilde = mapping(real(a'))';

% FOM test parameter

tstart = cputime;
[t,q_test] = ode45(@(t,u)rhsWaveSpatial(t,u,kappa,advect(t)),t,u0);
fprintf("tcpu (FOM) : %f\n", cputime-tstart)

fprintf("error MGalerkin: %f\n", norm(qtilde-q_test,'fro')/norm(q_test,'fro'))

%% Plot solution in time
fig=figure(11);
fig.Position = [100 100 950 400];2*(rand(r,1)-0.5)*10
subplot(1,2,1)
h=imagesc(q(:,:,1)');
set(gca,'YDir','normal')
sgtitle({"training", "$\partial_t q + u(t) \partial_x q = 0$"},'interpreter','latex')
title('$u(t)=-2$', 'interpreter','latex')
shading flat
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
xticks([])
yticks([])
colormap(jet/1.5)

subplot(1,2,2)
imagesc(q(:,:,2)');
set(gca,'YDir','normal')
title('$u(t) = 2$','interpreter','latex')
shading flat
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
xticks([])
yticks([])
caxis([0,1])
colormap jet
cb = colorbar;
ax = gca();
cb.Position(1) = ax.Position(1) + ax.Position(3)+1e-2;
cb.Position(2) = ax.Position(2);
cb.Position(4) = ax.Position(4);
save_fig_tikz('imgs/1d-advection-train',fig)


fig=figure(21);
sgtitle({"test", "$\partial_t q + 5\sin(2\pi t/T) \partial_x q = 0$"},'interpreter','latex')
fig.Position = [100 100 950 400];
subplot(1,2,1)
imagesc(q_test);
set(gca,'YDir','normal')
title('FOM','interpreter','latex')
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
xticks([])
yticks([])
shading flat
colormap jet


subplot(1,2,2)
imagesc(qtilde);
set(gca,'YDir','normal')
title('ROM','interpreter','latex')
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
xticks([])
yticks([])
shading flat
colormap jet
ax = gca();
caxis([0,1])
cb = colorbar;
cb.Position(1) = ax.Position(1) + ax.Position(3)+1e-2;
cb.Position(2) = ax.Position(2);
cb.Position(4) = ax.Position(4);
save_fig_tikz('imgs/1d-advection-test',fig)
%print('-depsc2', '-loose', '../../figures/FFTWave2');

%%
for j=1:r
    % linear derivative terms
    Ur_x(:,j)=Dx*Ur(:,j);
end

if 0
    Mass = @(t,a) Ur'*diag(dfront(Ur*a).^2)*Ur;
    Lin = @(a,c) -c*Ur'*diag(dfront(Ur*a).^2)*Ur_x;
    rom_ode = @(t,a) Lin(a,advect(t))*a;
    opts = odeset('Mass',Mass, 'MStateDependence','strong','RelTol',1e-5,'AbsTol',1e-4);

    tstart = cputime;
    [t_,a] = ode15s(rom_ode,t,a0, opts);
    fprintf("tcpu (ROM) : %f\n", cputime-tstart)
else
    %Mass = @(t,a) Ur'*Ur; % is the identiy
    UrtimesUrx=(Ur)'*(Ur_x);
    Lin = @(a,c) -c*UrtimesUrx;
    rom_ode = @(t,a) Lin(a,advect(t))*a;
    tstart = cputime;
    [t_,a] = ode45(rom_ode,t,a0);
    tcpu_rom = cputime-tstart;
    tstart = cputime;
    [t_,q_test] = ode45(@(t,u)rhsWaveSpatial(t,u,kappa,advect(t)),t,u0);
    tcpu_fom = cputime-tstart
    fprintf("speedup: %f\n",tcpu_fom/tcpu_rom )
end

%[t,a] = ode15s(rom_ode,t,a0);
subplot(1,2,1)
pcolor(q_test);
shading flat
colormap jet


qtilde = mapping(real(a'))';
subplot(1,2,2)
pcolor(qtilde);
shading flat
colormap jet
xticks([])
yticks([])
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
fprintf("error UrDxUr: %f\n", norm(qtilde-q_test,'fro')/norm(q_test,'fro'))

%%

rhs = @(t,a) (jac(a)'*jac(a))\jac(a)'*rhsWaveSpatial(t,mapping(a),kappa,advect(t));
[t,a2] = ode45(rhs,t,a0);
qtilde = mapping(real(a2'))';
fprintf("error EECSW: %f\n", norm(qtilde-q_test,'fro')/norm(q_test,'fro'))


%%
clear U,clear S, clear V, clear Ur, clear Ur_x


% convergence study:

advect = @(time) 5* sin(2*pi * time/t(end));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tstart = cputime;
Nrepeat = 10;
 for repeat = 1:Nrepeat
[t,q_test] = ode45(@(t,u)rhsWaveSpatial(t,u,kappa,advect(t)),t,u0);
 end
tcpu_fom = (cputime-tstart)/Nrepeat;
fprintf("tcpu (FOM) : %f\n", tcpu_fom)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nrepeat = 1000;
qmat = reshape(q,N,[]);
ir = 0;
ranks = [2:15];
for r = ranks
    ir = ir + 1;
    phi = FTR_direct(qmat, front, r, 3000);
    phi_mat = reshape(phi,N,[]);
    [U,S,V] = svds(phi_mat,r);
    Ur = U;
    ampl = S*V';
    mapping = @(a) front(Ur*a);
    %jac =  @(a) diag(dfront(Ur*a))*Ur;
    jac = @(a)  diff_mapping(mapping,a,r);
    error_FTR(ir) =  norm(qmat-mapping(ampl),'fro')/norm(qmat,'fro');
    %%%%%%%%%%%%%%%%%%%
    % FTR - Mgalerkin
    %%%%%%%%%%%%%%%%%%%
    a0 = ampl(:,1);
    rhs = @(t,a) jac(a)\rhsWaveSpatial(t,mapping(a),kappa,advect(t));
    [t,a] = ode45(rhs,t,a0);
    qtilde = mapping(real(a'))';
    error_MGFTR(ir) = norm(qtilde-q_test,'fro')/norm(q_test,'fro');

    %%%%%%%%%%%%%%%%%%%%
    % FTR - Galerkin
    %%%%%%%%%%%%%%%%%%%
    Ur_x=[];
    for j=1:r
        % linear derivative terms
        Ur_x(:,j)=Dx*Ur(:,j);
    end
    UrtimesUrx=(Ur)'*(Ur_x);
    Lin = @(a,c) -c*UrtimesUrx;
    rom_ode = @(t,a) Lin(a,advect(t))*a;

    tstart = cputime;
    for repeat = 1:Nrepeat
        [t_,a] = ode45(rom_ode,t,a0);
    end
    tcpu_rom_GFTR(ir) = (cputime-tstart)/Nrepeat;

    qtilde = mapping(real(a'))';
    error_GFTR(ir) = norm(qtilde-q_test,'fro')/norm(q_test,'fro');

    %%%%%%%%%%%%%%%%%%%%
    % POD - Galerkin
    %%%%%%%%%%%%%%%%%%%
    [U,S,V] = svds(qmat,r);
    U_x=[];
    for j=1:r
        % linear derivative terms
        U_x(:,j)=Dx*U(:,j);
    end
    DxrPOD=(U)'*(U_x);
    LinPOD = @(a,c) -c*DxrPOD;
    rom_ode_POD = @(t,a) LinPOD(a,advect(t))*a;

    tstart = cputime;
    for repeat = 1:Nrepeat
        [t_,a] = ode45(rom_ode_POD,t,a0);
    end
    tcpu_rom_GPOD(ir) = (cputime-tstart)/Nrepeat;
    qtilde = mapping(real(a'))';
    error_GPOD(ir) = norm(qtilde-q_test,'fro')/norm(q_test,'fro');
    error_POD(ir) =  norm(qmat-U*S*V','fro')/norm(qmat,'fro');
end

%%
speedupPOD = tcpu_fom./tcpu_rom_GPOD;
speedupFTR = tcpu_fom./tcpu_rom_GFTR;
fig = figure(22)
bar(ranks,speedupPOD,0.2)
%labelpoints(speedupPOD, error_GPOD, compose("(%d)",2:15),"N")


hold on
bar(ranks,speedupFTR,0.2)
%grid on
xlabel("degrees of freedom $r$","Interpreter","latex")
ylabel("speedup")
legend("POD Galerkin","FTR Galerkin")
legend boxoff
save_fig_tikz('imgs/1d-advection-test-speedup-vs-rank',fig)

%%
fig=figure(23)
semilogy([2:15],error_FTR,'o--')
hold on
semilogy([2:15],error_GFTR,'*-')
semilogy([2:15],error_MGFTR,'x-.')
semilogy([2:15],error_POD,'<--')
semilogy([2:15],error_GPOD,'>:')
xlabel("degrees of freedom $r$","Interpreter","latex")
ylabel("relative error")
legend boxoff
legend("FTR offline","FTR Galerkin","FTR manifold Galerkin","POD offline", "POD Galerkin","location","southwest")
save_fig_tikz('imgs/1d-advection-test-errors',fig)
