function [Pre_Labels,Outputs] = BR(train_data,train_target,test_data)
%BR Binary Relevance with linear Ridge Regression
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%           Outputs          An L x Nt output label matrix, each column is a label confidence array
%
%  [1] M.R. Boutel et al. Learning multi-label scene classification. Pattern Recognition, 2004.

%% Ridge parameter
lambda = 0.1;

%% Ridge Regression
ww = ridgereg(train_target',train_data,lambda);
Outputs = [ones(size(test_data,1),1) test_data] * ww;
Outputs = Outputs';
Pre_Labels = round(Outputs);

%% Regularize the predicted label set
Pre_Labels(Pre_Labels>1) = 1;
Pre_Labels(Pre_Labels<1) = 0;
   
end

