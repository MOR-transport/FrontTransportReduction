function dfda = diff_mapping(mapping,a,max_rank)
    fom0 = mapping(a);
    eps = 0.001;
    a_temp=a;
    for r = 1:max_rank
        a_temp(r) = a_temp(r) + eps;
        fom1 = mapping(a_temp);
        jac(:,r) = (fom1-fom0)/eps;
        a_temp(r)=a(r);
    end
    dfda = jac;

