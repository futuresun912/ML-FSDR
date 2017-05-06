function [Yt,Conf] = BR(X,Y,Xt)
%BR Binary Relevance with linear Ridge Regression
%       Input:
%           X     An N x D data matrix, each row denotes a sample
%           Y     An L x N label matrix, each column is a label set
%           Xt    An Nt x D test data matrix, each row is a test sample
%       Output:
%           Yt    An L x Nt predicted label matrix, each column is a predicted label set
%           Conf  An L x Nt output label matrix, each column is a label confidence array
%
%  [1] M.R. Boutel et al. Learning multi-label scene classification. Pattern Recognition, 2004.

%% Ridge parameter
lambda = 0.1;

%% Add bias into the data
X  = [ones(size(X,1),1),X];
Xt = [ones(size(Xt,1),1),Xt]; 

%% Ridge Regression
W    = Y * X / (X'*X+lambda*eye(size(X,2)));
Conf = W  * Xt';
Yt   = round(Conf);

%% Regularize the predicted label set
Yt(Yt>1) = 1;
Yt(Yt<1) = 0;
   
end