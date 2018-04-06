function D = kernel_embedding_D(data, data_sr)

%data is a L * m_x matrix
%data_sr is a L * m_x matrix
%feature_type is a m_x * 1 vector

%output K is a 3d array (L,L,m_x)
len1 = size(data,1); %number of rows in data
len2 = size(data_sr,1); %number of rows in data_sr

num_of_feature = size(data,2); 
tempD = zeros(len1,len2,num_of_feature);

for t = 1:num_of_feature
    dist = pdist2(data(:,t), data_sr(:,t));
    
    tempD(:,:,t) = dist;
end

D = tempD;

end