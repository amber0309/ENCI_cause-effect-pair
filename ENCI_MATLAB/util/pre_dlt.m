function dlt = pre_dlt(XY, Z, nspl)
%{
inputs:
XY    - cell of matrices
Z     - cell of matrices
nspl  - number of pts used to estimate delta

output:
dlt   - 1 by dim matrix
%}


dim = size(XY{1}, 2);
dimz = size(Z{1}, 2);

xyall = cat(1,XY{:});

if dimz ~= 1
    xyall = rand_perm_row(xyall);
    zall = xyall;
else
    zall = cat(1, Z{:});
    xyall = rand_perm_row(xyall);
    zall = rand_perm_row(zall);
end

dlt = zeros(1, dim);
if size(xyall, 1) > nspl
    n = nspl;
else
    n = size(xyall, 1);
end

for j = 1:dim
    if dimz == 1
        dlt(1,j) = median_dist(xyall(1:n, j), zall(1:n, 1));
    else
        dlt(1,j) = median_dist(xyall(1:n, j), zall(1:n, j));
    end
end

end