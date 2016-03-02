function eigenvecs = pca_fun(X, d)

% Implementation of PCA
% input:
%   X - N*D data matrix, each row as a data sample
%   d - target dimensionality, d <= D
% output:
%   eigenvecs: D*d matrix

Cx = X' * X ./ size(X,1);
[U, S, V] = svd(Cx);
eigenvecs = U(:, 1 : d);

%
% usage:
%   eigenvecs = pca_fun(X, d);
%   projection = X*eigenvecs;
%

end