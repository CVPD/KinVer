function S = kuncheva_stability(featidx,d)
%  kuncheva Stability index r

k = size(featidx,2);
q = size(featidx,1);

r = NaN(q,q);
% kuncheva index r
for n = 1:q-1
    for m = n+1:q
        r(n,m) = length(intersect(featidx(n,:),featidx(m,:)));
    end
end

A = (r-(k^2/d))./(k-(k^2/d));
A(isnan(A)) = 0;
S = 2.*sum(A(:))./(q*(q-1));
