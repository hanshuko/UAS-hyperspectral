%% gpml_pipeline.m
% Minimal GPML pipeline: standardize -> univariate screen -> PCA -> ARD GP fit
% Requires GPML toolbox on MATLAB path (https://gaussianprocess.org/gpml/code/matlab/doc/)

clc; clear; close all;
rng(0);  % for reproducibility

% -----------------------
% USER: load your data here
% X: N x D  matrix (N samples, D=125)
% y: N x 1  vector (targets)
% -----------------------
%Load data from CSV
soilMoistureMatrix = readtable("soilmoisture_dataset.xlsx");

%Split data into 2 matrices, one for hyperspectral bands by pixel and
%one for moisture responses
moisture = soilMoistureMatrix(2:end,3);
signals = soilMoistureMatrix(2:end,5:end);
moisture = table2array(moisture); signals = table2array(signals);

%Make indexing array by selecting random pixels, approximately 70% of
%total number of pixels
totalPix = 679; trainPix = 475; numBands = 125;
index = randperm(totalPix,trainPix);

%Extract random pixels and their corresponding moisture content into their 
%own matrix
X = zeros(trainPix,numBands); y = zeros(trainPix,1);
for s = 1:trainPix
    X(s,:) = signals(index(s),:);
    y(s) = moisture(index(s));
end

% standardize inputs & outputs
Xmu = mean(X,1);
Xs  = std(X,[],1);
Xs(Xs==0) = 1;             % guard against zero-variance features
Xstd = (X - Xmu) ./ Xs;

ymu = mean(y);
ys  = std(y);
if ys == 0, ys = 1; end
ystd = (y - ymu) ./ ys;

[N, D] = size(Xstd);

% -----------------------
% Univariate screening (Pearson & Spearman)
% -----------------------
pearson_r = zeros(D,1);
spearman_r = zeros(D,1);
for d = 1:D
    pearson_r(d) = corr(Xstd(:,d), ystd, 'Type', 'Pearson');
    spearman_r(d) = corr(Xstd(:,d), ystd, 'Type', 'Spearman');
end

[~, idx_p] = sort(abs(pearson_r), 'descend');
[~, idx_s] = sort(abs(spearman_r), 'descend');

fprintf('Top 10 features by |Pearson r|: %s\n', mat2str(idx_p(1:min(10,D))'));
fprintf('Top 10 features by |Spearman r|: %s\n', mat2str(idx_s(1:min(10,D))'));

% -----------------------
% PCA projection (visual diagnostics)
% -----------------------
[coeff, score, latent] = pca(Xstd, 'NumComponents', min(10,D));
% plot y vs first 2 PCs (user can run these lines interactively)
figure;
subplot(1,2,1);
scatter(score(:,1), ystd, 30, 'filled'); xlabel('PC1'); ylabel('y (std)'); title('y vs PC1');
subplot(1,2,2);
scatter(score(:,2), ystd, 30, 'filled'); xlabel('PC2'); ylabel('y (std)'); title('y vs PC2');

% -----------------------
% GP: Mean = linear+const, Cov = SE-ARD + white noise, Likelihood = Gaussian
% (common sensible starter)
% -----------------------
addpath('gpml');              % adjust path if needed
startup;                      % run GPML startup if not already run

meanfunc = {@meanSum, {@meanLinear, @meanConst}};  % affine mean
covfunc  = {@covSum, {@covSEard, @covNoise}};      % ARD RBF + white-noise diag
likfunc  = @likGauss;
infmethod = @infExact;        % exact inference for Gaussian likelihood

% Hyperparameter initialization:
% - covSEard needs D lengthscales + 1 signal variance -> (D+1) entries
% - covNoise needs 1 noise variance -> 1 entry
% So hyp.cov length = (D+1) + 1 = D+2
% - meanLinear needs D weights, meanConst needs 1 -> hyp.mean length = D+1
hyp = struct();
hyp.mean = zeros(D+1,1);          % init linear weights and constant to zero
hyp.cov  = zeros(D+2,1);          % init log-lengthscales, log-sf, log-noise
hyp.lik  = log(0.1);              % init Gaussian noise std (log scale)

% Optionally - better init: set lengthscales ~ 1 (since inputs standardized)
% hyp.cov(1:D) = log(1);          % log lengthscales
% hyp.cov(D+1) = log(1);          % log signal std
% hyp.cov(D+2) = log(0.1);        % log noise std

% Train (optimize hyperparameters)
% minimize uses the 'minimize' helper from GPML
num_restarts = 5; % can increase (slower)
best_hyp = hyp;
best_nlZ = inf;
for r = 1:num_restarts
    % small random jitter to avoid local minima
    init = hyp;
    init.cov = init.cov + 0.1*randn(size(init.cov));
    init.mean = init.mean + 0.01*randn(size(init.mean));
    init.lik = init.lik + 0.01*randn;
    try
        opt = minimize(init, @gp, -200, infmethod, meanfunc, covfunc, likfunc, Xstd, ystd);
    catch ME
        warning('minimization failed on restart %d: %s', r, ME.message);
        opt = init;
    end
    nlZ = gp(opt, infmethod, meanfunc, covfunc, likfunc, Xstd, ystd);
    if nlZ < best_nlZ
        best_nlZ = nlZ;
        best_hyp = opt;
    end
end
hyp = best_hyp;

% -----------------------
% Inspect ARD lengthscales (relevance)
% -----------------------
log_ells = hyp.cov(1:D);
ells = exp(log_ells);            % actual lengthscales (in standardized input-space)
signal_std = exp(hyp.cov(D+1));
noise_std = exp(hyp.lik);        % likGauss uses hyp.lik = log(sn)

% Smaller lengthscale => function varies rapidly with that input -> more relevant
[ell_sorted, ell_idx] = sort(ells);
fprintf('Top 12 most-relevant features (smallest lengthscales):\n');
disp(ell_idx(1:min(12,D))');

% optionally compute a "relevance" score (inverse lengthscale)
relevance = 1./ells;
[~, rel_idx] = sort(relevance,'descend');

% -----------------------
% Predict on training points (quick check) and plot predictions ± 2σ
% -----------------------
[Xmu_test, Xs_test] = deal(Xmu, Xs); % in case you want to standardize new Xsame way
[m, s2, fmu, fs2] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, Xstd, ystd, Xstd);
% convert back to original y scale
m_orig = m * ys + ymu;
sd_orig = sqrt(s2) * ys;

figure;
scatter(1:N, y, 'b'); hold on;
errorbar(1:N, m_orig, 2*sd_orig, 'r.');
legend({'data y', 'GP mean ± 2σ'}); title('GP fit on training set (for quick diagnostics)');
hold off;

% -----------------------
% Diagnostics & suggestions
% -----------------------
% - If many ells are very large (>>1), those inputs are likely irrelevant.
% - If residuals show input-dependent spread, consider heteroscedastic models (GPML supports likT and
%   approximations; see GPML docs).
% - If too slow (large N), consider FITC/SVGP approximations in GPML.
%
% -----------------------
% End of script
% -----------------------

%% Testing Section
%Initialize testing matrix containing all pixels not used in training
testPix = totalPix - trainPix;
test = zeros(testPix,1);
u = 1;

%Go through original pixel indexing matrix, add pixel to testing matrix
%if it is not present in the indexing matrix
for t = 1:totalPix
    present = find(index==t);
    if all(present == 0)
        test(u) = t;
        u = u + 1;
    end
end

inputs = zeros(testPix,numBands); actualResult = zeros(testPix,1);
for c = 1:testPix
    inputs(c,:) = signals(test(c),:);
    actualResult(c) = moisture(test(c));
end

%Standardize new results
inputsSTD = (inputs - Xmu) ./ Xs;
actualResultSTD = (actualResult - ymu) ./ ys;

[mT, s2T, fmuT, fs2T] = gp(hyp, infmethod, meanfunc, covfunc, likfunc, Xstd, ystd, inputsSTD);

error = rmse(actualResultSTD, mT);

%% Other Method for Comparison (Best model so far)

Mdl = fitrgp(Xstd,ystd,'KernelFunction','squaredexponential','FitMethod','sr');
testResult = zeros(testPix,1);
for a = 1:testPix
    testResult(a) = predict(Mdl,signals(test(a),:));
end

errorRQ = rmse(actualResultSTD,testResult);