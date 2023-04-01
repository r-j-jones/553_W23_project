
d = 10;
N = d;
r = 3;
k = 5;


oi = randsample(d,r);
oj = randsample(N,r);

ki = cat(1, oi, randsample(d,k-r));
kj = cat(1, oi, randsample(N,k-r));

intersect(oi,ki)