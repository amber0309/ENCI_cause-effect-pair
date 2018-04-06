function perm_mat = rand_perm_row(X)


random_x = X(randperm(size(X, 1)), :);
% [~,n] = size(X) ;
% idx = randperm(n) ;
% b = X;
% b(1,idx) = X(1,:);
perm_mat = random_x;

end