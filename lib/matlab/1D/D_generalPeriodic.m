function D  = D_generalPeriodic(N,h, koeffs)
% Expicit (derivate) matrix with  general coefficients
% Periodic Version
% D = D_leftOffen(N,h)
% N number of Points
% h = Delta x   

diagonalLow = -(length(koeffs)-1)/2 ;  
diagonalUp  = -diagonalLow ; 

D= sparse(N,N); 

for k =diagonalLow:diagonalUp
    D = D + koeffs(k -diagonalLow +1)*sparse(diag(ones(N-abs(k),1) , k)) ; 
    if k<0     
        D = D + koeffs(k -diagonalLow +1)*sparse(diag(ones(abs(k),1), N+k )  );  
    end
    if k>0     
        D = D + koeffs(k -diagonalLow +1)*sparse( diag(ones(abs(k),1), -N+k )  );  
    end
    
end


D= 1/(h)*D ; 

end
