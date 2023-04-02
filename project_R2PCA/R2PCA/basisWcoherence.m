function U = basisWcoherence(d,r,mu)
% ===== Auxiliary function to obtain a basis of a subspace within a =====
% ===== range of a specified coherence. This is done by increasing  =====
% ===== the magnitude of somerows of the basis U.                   =====
%
% INPUTS: 
%   d = # of rows in data matrix M and used for output U
%   r = rank of L low rank matrix
%   mu = coherence level
% OUTPUTS:
%   U = basis of subspace with specified coherence [i.e. coherence(U)=mu]

U = randn(d,r);
c = coherence(U);
i = 1;
it = 1;
while c<mu
    U(mod(i,d+1),:) = U(i,:)/(10^it);
    i = i+1;
    c = coherence(U);
    if i==d+1
        i = 1;
        it = it+1;
    end
end

end