function [tau_x, tau_y] = pre_tensor(XY)

N_grp = length(XY);

stand_xy = cell(1, N_grp);
for k = 1:N_grp
    stand_xy{k} = zscore(XY{k});
end

dlt_vec = pre_dlt(stand_xy, stand_xy, 1000);
tau_x = zeros(N_grp, 1);
tau_y = zeros(N_grp, 1);

for kidx = 1:N_grp
    xy = stand_xy{kidx};
    L = size(xy, 1);
    H = eye(L)-1/L*ones(L,L);
    
    x = xy(:,1);
    d_x = kernel_embedding_D(x, x);
    k_x_i = kernel_embedding_K(d_x, 1, dlt_vec(1, 1));
    tau_xi = trace( k_x_i * H )/L/L;
    tau_x(kidx, 1) = tau_xi;
    
    y = xy(:,2);
    d_y = kernel_embedding_D(y, y);
    k_y_i = kernel_embedding_K(d_y, 1, dlt_vec(1, 2));
    tau_yi = trace( k_y_i * H )/L/L;
    tau_y(kidx, 1) = tau_yi;
    
end

tau_x = tau_x - mean(tau_x);
tau_y = tau_y - mean(tau_y);

end