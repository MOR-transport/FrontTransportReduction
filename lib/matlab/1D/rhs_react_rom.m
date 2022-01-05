function dadt = rhs_react_rom(t,a,ddf,df,Psi,Psi_one, Psi_x,PPsi_xx, delta)
    
 b = (ddf(Psi*a).*(Psi_x*a).^2);
 A = diag(df(Psi*a))*Psi;
 x = A\b;
 dadt = PPsi_xx*a - 8/delta.^2*Psi_one +0*x;


end