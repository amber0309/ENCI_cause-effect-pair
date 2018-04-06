function order = ENCI_pair(XY)
%{
Input (1 or 2 input arguments)
XY      - the cell array of all groups of data, each group is a matrix
          where rows corresponds to i.i.d sample and columns
          corresponds to random variables.

Output
order   - the estimated causal direction
          if denote the first column by X and the second by Y, then
          1:  X --> Y
          -1: Y --> X

Usage
order = ENCI_pairs(X)

Shoubo (shoubo.sub AT gmail.com)
06/04/2018
%}

addpath('util');

% prepare the embedding of data groups
[tau_x, tau_y] = pre_tensor(XY);

% parameters for HSIC
params.sigx = -1;
params.sigy = -1;
alpha = 0.05;

% ----- compute residual and conduct HSIC test for x-->y
% regression option 1
% [~,~,resi_x2y] = regress(tau_y, [tau_x, ones(L, 1)]);

% regression option 2
mdl_x2y = fitlm(tau_x, tau_y);
resi_x2y = mdl_x2y.Residuals.Raw;

[thresh_x2y, testStat_x2y, ~] = hsicTestGamma(tau_y, resi_x2y, alpha, params);
r_x2y = testStat_x2y / thresh_x2y;

% ----- compute residual and conduct HSIC test for y-->x
% regression option 1
% [~,~,resi_y2x] = regress(tau_x, [tau_y, ones(L, 1)]);

% regression option 2
mdl_y2x = fitlm(tau_y, tau_x);
resi_y2x = mdl_y2x.Residuals.Raw;

[thresh_y2x, testStat_y2x, ~] = hsicTestGamma(tau_x, resi_y2x, alpha, params);
r_y2x = testStat_y2x / thresh_y2x;

% ----- conclude caual direction
% fprintf('r_x2y = %f\n', r_x2y);
% fprintf('r_y2x = %f\n', r_y2x);

if r_x2y < r_y2x
    fprintf('The causal direction is X --> Y\n');
    order = 1;
else
    fprintf('The causal direction is Y --> X\n');
    order = -1;
end

end