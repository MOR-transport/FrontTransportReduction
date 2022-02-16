function rhs_hyp = rhs_rom_move(a,jac,rhs,dfront,Ur, npoints)
  M = size(Ur,1);
  I = speye(M,M);
  %[~,n] = findpeaks(abs(dfront(Ur*a)),1:M);
  [~,n] = maxk(abs(dfront(Ur*a)),npoints);
  %[~,n] = mink(abs(Ur*a),npoints);
  P = I(n,:);
  rhs_hyp =(P*jac(a))\(P*rhs(a));
  %rhs_hyp =((P*jac(a))'*P*jac(a))\((P*jac(a))'*P*rhs(a));
end