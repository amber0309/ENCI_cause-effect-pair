function MM = median_dist(S1, S2)

num_of_feature = size(S1,2); 
tempMM = zeros(1,num_of_feature); %store median of each column

for t = 1:num_of_feature
    dist = pdist2(S1(:,t), S2(:,t));
    
    tempMM(1,t) = median(dist(:));
end

MM = tempMM;


end