function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%

N = size(train_data, 1);
D = size(train_data, 2);

%% solve alpha in dual problem using quadprog function
H = (train_label * train_label') .* (train_data * train_data');
f = - ones(N, 1);
A = [];
b = [];
Aeq = train_label';
beq = 0;
lb = zeros(N, 1);
ub = C * ones(N, 1);
x0 = [];
options = optimset ('Algorithm', 'interior-point-convex');
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);

%% compute the value for w
y_matrix = repmat(train_label, 1, D);
w = (alpha' * (y_matrix .* train_data))';

%% conpute the value for b
x_positive = train_data(train_label == 1, :);
positive_d = sort(w' * x_positive');
min_x = positive_d(1);
x_negative = train_data(train_label == -1, :);
negative_d = sort(w' * x_negative', 'descend');
max_x = negative_d(1);

b = - (min_x + max_x) / 2;


end





