function Yt = FSDR(X,Y,Xt,opts)
% FSDR: A wrapper of Feature Space Dimension Reduction algorithms for MLC
%
%    Description
%
%       Input:
%           X       An N  x F training data matrix
%           Y       An L  x N label matrix
%           Xt      An Nt x F test data matrix
%           opts:   Input parameters
% 
%       Output:
%           Yt      An L x Nt predicted label matrix
%
% References:
% PCA:  Peason, K. (1901). On lines and planes of closest fit to systems of point in space. Philosophical Magazine, 2(11), 559-572.
% LPP:  He, X., & Niyogi, P. (2003, December). Locality preserving projections. In NIPS (Vol. 16, No. 2003).
% NPE:  He, X., Cai, D., Yan, S., & Zhang, H. J. (2005, October). Neighborhood preserving embedding. In Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on (Vol. 2, pp. 1208-1213). IEEE.
% MLSI: Yu, K., Yu, S., & Tresp, V. (2005, August). Multi-label informed latent semantic indexing. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 258-265). ACM.
% MLDA: Wang, H., Ding, C., & Huang, H. (2010, September). Multi-label linear discriminant analysis. In European Conference on Computer Vision (pp. 126-139). Springer Berlin Heidelberg.
% MDDM: Zhang, Y., & Zhou, Z. H. (2010). Multilabel dimensionality reduction via dependence maximization. ACM Transactions on Knowledge Discovery from Data (TKDD), 4(3), 14.
% HSL:  Sun, L., Ji, S., & Ye, J. (2008, August). Hypergraph spectral learning for multi-label classification. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 668-676). ACM.
% CCA:  Hotelling, H. (1936). Relations between two sets of variates. Biometrika, 28(3/4), 321-377.
% OPLS: Worsley, K. J., Poline, J. B., Friston, K. J., & Evans, A. C. (1997). Characterizing the response of PET and fMRI data using multivariate linear models. NeuroImage, 6(4), 305-319.
 
%% Set parameters
dim    = opts.dim;
gamma  = opts.gamma;
beta   = opts.beta;
alg    = opts.alg;
opt_w  = opts.opt_w;
[numN,numF] = size(X);

%% Center the data and target matrices
tmpY   = Y';
Xmean  = mean(X,1);
tmpX   = bsxfun(@minus,X,Xmean);
tmpXt  = bsxfun(@minus,Xt,Xmean);
tmpY   = bsxfun(@minus,tmpY,mean(tmpY,1));

%% Solve the optimization problem: A*U = B*U*D
switch alg
    case 'pca'  % PCA
        Sxx   = tmpX' * tmpX;
        A     = Sxx;
        B     = [];
    case 'mddm' % MDDM
        Sxy   = tmpX' * tmpY;
        A     = Sxy * Sxy';
        B     = [];
    case 'mlsi' % MLSI
        Sxx   = tmpX' * tmpX;
        Sxy   = tmpX' * tmpY;
        A     = (1-beta).*Sxx*Sxx + beta.*Sxy*Sxy';
        B     = Sxx + gamma.*speye(numF);
    case 'opls' % OPLS
        Sxx   = tmpX' * tmpX;
        Sxy   = tmpX' * tmpY;
        A     = Sxy * Sxy';
        B     = Sxx + gamma.*speye(numF);
    case 'cca'  % CCA
        tmpY  = tmpY(:,any(tmpY,1));  
        Sxx   = tmpX' * tmpX;
        Sxy   = tmpX' * tmpY;
        Syy   = tmpY' * tmpY;
        A     = Sxy / Syy * Sxy';
        B     = Sxx + gamma.*speye(numF);
    case 'hsl'  % HSL
        W     = constructW(Y',opt_w);
        D     = sparse(1:numN,1:numN,sum(W,1),numN,numN);
        Sxx   = tmpX' * tmpX;
        A     = tmpX' * (D.^.5*W*D.^.5) * tmpX;
        B     = Sxx + gamma.*speye(numF);
    case 'lpp'  % LPP
        W     = constructW(X,opt_w);
        D     = sparse(1:numN,1:numN,sum(W,1),numN,numN);
        A     = tmpX' * W * tmpX;
        B     = tmpX' * D * tmpX + gamma.*speye(numF);
    case 'npe'  % NPE
        W     = constructW(X,opt_w);
        M     = speye(numN) - W;
        M     = M'*M;
        Sxx   = tmpX' * tmpX;
        A     = tmpX' * M * tmpX;
        B     = Sxx + gamma.*speye(numF);
    case 'mlda' % MLDA
        newY  = Y(any(Y,2),:);  
        C     = 1 - pdist(newY,'cosine');
        C     = sparse(squareform(C));
        C(logical(eye(size(C)))) = 1;
        Yc    = newY' * C;
        Z     = bsxfun(@rdivide,Yc,sum(newY,1)');
        m     = sum(Yc'*X,1) ./ sum(sum(Yc));
        tmpX  = X - ones(numN,1)*m;
        tmpXt = Xt - ones(size(Xt,1),1)*m;
        numL  = size(newY,1);
        W     = sparse(1:numL,1:numL,sum(Z,1).^-1,numL,numL);     
        L     = sparse(1:numN,1:numN,sum(Z,2),numN,numN); 
        Sxz   = tmpX' * Z;
        A     = Sxz * W * Sxz';
        B     = tmpX' * L * tmpX + gamma.*speye(numF);
end
[U,~] = eigs(A,B,dim);

%% Encode the data matrices
tmpX  = tmpX * U;
tmpXt = tmpXt * U;

%% Build the multi-label classifier
Yt    = BR(tmpX,Y,tmpXt);

end