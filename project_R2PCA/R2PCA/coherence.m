% ===== Auxiliary function to compute coherence paraemter =====
function mu = coherence(U)
P = U/(U'*U)*U';
[d,r] = size(U);

Projections = zeros(d,1);
for i=1:d
    ei = zeros(d,1);
    ei(i) = 1;
    Projections(i) = norm(P*ei,2)^2;
end

mu = d/r * max(Projections);
end
