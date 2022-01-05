clear all, close all, clc
addpath("lib")
%% Define spatial domain
delta_train = [0.1,1];              % Wave speed
d_test = [0.3];              % Wave speed
L = 30;             % Length of domain
N =1000;           % Number of discretization points
dx = L/N;
T = 1;
r=4;
method = "nAPG"; % choose APG for adjoint petrov galerkin, else Galerkin
x = -L/2:dx:L/2-dx; % Define x domain

front = @(x) 0.5*(1-tanh(x/2));
dfront = @(x) -0.25./(cosh(x/2).^2);
ddfront = @(x) 0.25*sinh(x/2)./(cosh(x/2).^3);

% Define discrete wavenumbers
kappa = (2*pi/L)*[-N/2:N/2-1];
kappa = fftshift(kappa');    % Re-order fft wavenumbers

h = x(2)-x(1);
koefDX = [-1/60 	3/20 	-3/4 	0 	3/4 	-3/20 	1/60] ;
%koefDX = [1/280, 	-4/105 ,	1/5, 	-4/5, 	0 ,	4/5, 	-1/5, 	4/105 ,	-1/280 ] ;
%koefDXX=  [ 0 1 -2 1  0 ] ;
koefDXX= [-1/12 	4/3 	-5/2 	4/3 	-1/12] ;
%koefDXX = [-1/560 	8/315 	-1/5 	8/5 	-205/72 	8/5 	-1/5 	8/315 	-1/560];
%Dxx  = D_generalPeriodic(N(1), 2*h^2  ,koefDXX  ) ;
%kappa = Dxx;
% Simulate in Fourier frequency domain
rhs_fom = @(t,u,kappa,d) rhs_react_fft(t,u,kappa,d);
t =linspace(0,T,101);

[X,T] = meshgrid(x,t);

phi = zeros(N,length(t),length(delta_train));
q = zeros(N,length(t),length(delta_train));
i=0;

for d = delta_train
    u0 = front((abs(x)-2)/d*2);
    i = i +1;
    %phi_temp =(abs(X)-2*T/d-2)/d;
    %phi(:,:,i) = phi_temp';
    % Alternatively, simulate in spatial domain
    
    [t,u] = ode45(@(t,u)rhs_fom(t,u,kappa,d),t,u0);
    q(:,:,i) = u';
end
usol =@(d) front((abs(X)-2*T/d-2)/d*2) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ROM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phi = FTR_direct(reshape(q,N,[]), front, r, 3000);
[U,S,V] = svds(reshape(phi,N,[]),r);
Ur = U;
mapping = @(a) front(Ur*a);
%jac =  @(a) diag(dfront(Ur*a))*Ur;
jac = @(a)  diff_mapping(mapping,a,r);
%jac_inv =@(a)  Ur'*diag(1./(dfront(Ur*a)+1e-13))
phi0 = (abs(x)-2)/d_test;

%% ROM test parameter
fprintf("ROM number modes: %d \n",r)
u0 = front((abs(x)-2)/d_test*2);

a = S*V';
fitfun= @(a) u0-mapping(a);
a0 = a(:,1);
%a0 = lsqnonlin(fitfun,a(:,1))

if method ~= "APG" 
   rhs_rom = @(t,a) jac(a)\rhs_fom(t,mapping(a),kappa,d_test);
else
    fprintf("using adjoint petrov galerkin\n")
    rhs = @(time,u) rhs_fom(time,u,kappa,d_test);
    lin_rhs = @(time,u,du) jac_rhs_fom(time,u,du,kappa,d_test);
    rhs_rom = @(t,a) APG_rhs(t,a, rhs,lin_rhs,mapping,jac, 1);
end
tstart = cputime;
[t_,atilde] = ode45(rhs_rom,t,a0);
qtilde = mapping(real(atilde'))';
fprintf("tcpu (ROM) : %f\n", cputime-tstart)

% FOM test parameter
% initial condition

tstart = cputime;
[t,q_test] = ode45(@(t,u)rhs_fom(t,u,kappa,d_test),t,u0);
fprintf("tcpu (FOM) : %f\n", cputime-tstart)
fprintf("rel error (FOM-ROM)  : %f\n",norm(qtilde-q_test,'fro')/norm(q_test,'fro'))
%%
figure(49)
for it = 1: length(t)
    L2err(it) = norm(qtilde(it,:)-q_test(it,:),'fro')/ norm(q_test(it,:),'fro');
    plot(q_test(it,:))
    hold on
    plot(qtilde(it,:))
    hold off
    pause(0.01)
end

figure(89)
plot(L2err)
xlabel("time")
ylabel("error")
%% Plot solution in time
figure(10)
subplot(2,2,1)
h=pcolor(X,T,q(:,:,1)');
title(['train data $\delta=',num2str(delta_train(1)),'$'], 'interpreter','latex')
shading flat
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
colormap(jet/1.5)

subplot(2,2,2)
pcolor(X,T,q(:,:,2)');
title(['train data $\delta=',num2str(delta_train(2)),'$'], 'interpreter','latex')
shading flat
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
colormap jet

subplot(2,2,3)
pcolor(X,T,q_test);
title(['FOM $\delta_{\mathrm{test}}=',num2str(d_test),'$'],'interpreter','latex')
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
shading flat
colormap jet

subplot(2,2,4)
pcolor(X,T,qtilde);
title(['ROM $\delta_{\mathrm{test}}=',num2str(d_test),'$'],'interpreter','latex')
xlabel('$x$',  'interpreter','latex')
ylabel('$t$', 'interpreter','latex')
shading flat
colormap jet
%print('-depsc2', '-loose', '../../figures/FFTWave2');
print('-depsc2','-r600', 'comparison');
%%
% Hyperreduction

it = 10;
M = N;
err_min = 1;
cnum = 1e10
nsample = 200;
qmat = reshape(q,N,[]);
fom_rhs = rhs_fom(t,qmat(:,it),kappa,delta_train(1));
I = speye(M,M);
for i = 1:1000
    n2 = randsample(M,nsample);
    P= I(n2,:);
    rhs_hyp = @(t,a) (P*jac(a))\(P*rhs_fom(t,mapping(a),kappa,delta_train(1)));
    err= norm(jac(a(:,it))*rhs_hyp(t,a(:,it))-jac(a(:,it))*rhs_rom(t,a(:,it)))/norm(jac(a(:,it))*rhs_rom(t,a(:,it)));
    err= norm(jac(a(:,it))*rhs_hyp(t,a(:,it))-fom_rhs)/norm(jac(a(:,it))\fom_rhs);
    for it = 1:size(t)
        carray(it) = cond(P*jac(a(:,it)));
    end
    if cnum > sum(carray)
        nsave2 = n2;
        cnum = sum(carray)
    end
    if err <err_min
        err_min = err;
        nsave = n2;
    end
end

err_min
qtilde = mapping(a(:,it));
plot(qtilde)
hold on
plot(nsave,qtilde(nsave),'*')


%% hyper ROM

fprintf("ROM number modes: %d \n",r)
a = S*V';
a0 =a(:,1); %Ur'*phi0'; % a(:,1)
Nsample = 100;
if method ~= "APG"
    I = eye(N,N);
    n2 = randsample(N,Nsample);
    P= I(n2,:);
    %rhs_rom = @(t,a) (P*jac(a))\(P*rhs_fom(t,mapping(a),kappa,d_test));
    rhs_rom = @(t,a) rhs_rom_move(a,jac,@(a) rhs_fom(t,mapping(a),kappa,d_test),dfront,Ur, Nsample);
    
    %rhs_rom = @(t,a) jac(a)\rhs_fom(t,mapping(a),kappa,d_test);
else
    fprintf("using siadjoint petrov galerkin\n")
    rhs = @(time,u) rhs_fom(time,u,kappa,d_test);
    lin_rhs = @(time,u,du) jac_rhs_fom(time,u,du,kappa,d_test);
    rhs_rom = @(t,a) APG_rhs(t,a, rhs,lin_rhs,mapping,jac, 1);
end
tstart = cputime;
[t_,atilde] = ode45(rhs_rom,t,a0);
qtilde = mapping(real(atilde'))';
fprintf("tcpu (ROM) : %f\n", cputime-tstart)

% FOM test parameter
% initial condition
%u0 = front((abs(x)-2)/d_test);
tstart = cputime;
[t,q_test] = ode45(@(t,u)rhs_fom(t,u,kappa,d_test),t,u0);
fprintf("tcpu (FOM) : %f\n", cputime-tstart)
fprintf("rel error (FOM-ROM)  : %f\n",norm(qtilde-q_test,'fro')/norm(q_test,'fro'))


%% DEIM
NL = [rhs_fom(t,q(:,:,1),kappa,delta_train(1)),rhs_fom(t,q(:,:,2),kappa,delta_train(2))];
%NL = [rhs_fom(t,front(phi(:,1:101)),kappa,delta_train(1)),rhs_fom(t,front(phi(:,102:end)),kappa,delta_train(2))];
[XI,S_NL,W]=svd(NL,0);
[Xi_max,nmax]=max(abs(XI(:,1)));
XI_m=XI(:,1);
z=zeros(N,1);
P=z; P(nmax)=1;

% DEIM points 2 to r
for j=2:r
    c=(P'*XI_m)\(P'*XI(:,j));
    res=XI(:,j)-XI_m*c;
    [Xi_max,nmax]=max(abs(res));
    XI_m=[XI_m,XI(:,j)]; 
    P=[P,z]; P(nmax,j)=1;  
end
    
% Qdeim     
[~,~,pivot]=qr(NL.');
P_deim=pivot(:,1:r);

[phisort,I] = sort(phi,1);
NLsort = NL(I);
[XI,S_NL,W]=svds(NLsort,3);
NLsorttilde = XI*S_NL*W';

