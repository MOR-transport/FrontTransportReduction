function dadt = APG_rhs(time, romstate, rhs, lin_rhs, mapping, jac_mapping, C)
    % Adjoint petrov galerkin
    % algorithm 1 in https://arxiv.org/pdf/1810.03455.pdf
    if (nargin < 7 )
        C = 0.2;
    end
    fomstate = mapping(romstate);
    R = rhs(time,fomstate);
    jac = jac_mapping(romstate);
    ProjR = jac\R;           % this is the "normal galerkin RHS"
    ProjR = jac*ProjR;       % Projection of the RHS
    OrthProjR= R - ProjR;               % orthogonal projection
    JacRHS = lin_rhs(time,fomstate,OrthProjR); 
    tau = C/abs(eigs(jac\lin_rhs(time,fomstate,jac),1,'lm'));
    dadt = jac\(ProjR + tau*JacRHS);
    
end