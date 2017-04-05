%% Make experiments repeatedly
rng('default');

%% Add pathes containing supporting functions
addpath('data','func','eval');

%% Load a FSDR algorithm and a dataset
opts.alg    = 'pca';  % pca, mddm, mlsi, cca, mlda, opls, hsl, lpp, npe
dataset     = 'enron';
load([dataset,'.mat']);

%% Scale data into [0,1] in coloumn-wise
data = data(:,any(data,1)); 
minX = min(data,[],1);
diff = max(data,[],1) - minX;
data = bsxfun(@minus,data,minX);
data = bsxfun(@rdivide,data,diff);

%% Set parameters
opts.dim    = 100;    % dimensionality of the feature subspace
opts.gamma  = 1;
opts.beta   = 0.5;
opt_w.k     = 10;
opt_w.NeighborMode = 'KNN';
opt_w.WeightMode   = 'HeatKernel';
opts.opt_w  = opt_w;

%% Perform n-fold cross validation
numFold = 5; 
indices = crossvalind('Kfold',size(data,1),numFold);
Results = zeros(5,numFold);
for i = 1:numFold
    disp(['Fold ',num2str(i)]);
    test  = (indices==i); 
    train = ~test;  
    tic; Pre_Labels = FSDR(data(train,:),target(:,train),data(test,:),opts);
    Results(1,i) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test));
    Results(2:end,i) = [ExactM,HamS,MacroF1,MicroF1];
end
meanResults = squeeze(mean(Results,2));
stdResults  = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],[dataset,'_',opts.alg],'Time ExactM HammingS MacroF1 MicroF1','Mean Std.');
