function y = kernel_embedding_K(dist, theta, delta)

%dist is a 3d array (i,j,k)
%theta is a constant
%delta is a 1 * k vector

%output y is a i * j matrix

Len = size(dist,3); %input is a 3d array, number 3 means we output the number of the third dimension
[m,n] = size(dist(:,:,1));
tempy = ones(m,n);

for i = 1:Len
    d = dist(:,:,i);%.^0.5;
    l = delta(1,i);
    % y1 = (1+sqrt(3)*d/l).*(exp(-sqrt(3)*d/l));%no need to modify, results are the same
    y2 = exp(-(d.^2)/(l.^2));
    
    tempy = tempy .* y2;
end
tempy = theta * tempy;
y = tempy;
