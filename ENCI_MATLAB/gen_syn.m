function xy = gen_syn(label, sample_size)
ncoeff = 1;
wt = rand(3,1) + 0.5;
wt = wt/sum(wt);

L1 = floor(wt(1, 1) * sample_size);
x1 = 0.3 * randn(L1, 1) - 1;
L2 = floor(wt(2, 1) * sample_size);
x2 = 0.3 * randn(L2, 1) - 1;
L3 = sample_size - L1 - L2;
x3 = 0.3 * randn(L3, 1);

x = [x1; x2; x3];
c = 0.4 * rand() + 0.8;
n = randn(sample_size, 1);

if label == 0
    y = 1 ./ (x.^2 + 1) .* n * ncoeff;
elseif label == 1
    y = sign(c*x) .* ((c*x).^2) .* n * ncoeff;
elseif label == 2
    n = - randn(sample_size, 1);
    y = cos(c * x .* n) .* n * ncoeff;
elseif label == 3
    y = sin( c * x) .* n * ncoeff;
elseif label == 4
    y = x.^2 .* n * ncoeff;
elseif label == 5
    y = (2* sin(x) + 2*cos(x)) .* n * ncoeff;
elseif label == 6
    y = 4 * (abs(x).^(0.5)) .* n * ncoeff;
end

xy = [x, y];
end