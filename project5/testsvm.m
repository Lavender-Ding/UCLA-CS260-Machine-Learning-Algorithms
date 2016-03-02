function accu = testsvm(test_data, test_label, w, b)
% Test linear SVM 
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  test_label: M*1 vector, each row as a label
%  w: feature vector 
%  b: bias term
%
% Output:
%  accu: test accuracy (between [0, 1])
%

N = size(test_data, 1);
D = size(test_data, 2);
b_vector = repmat(b, N, 1);
y_predict = zeros(N, 1);

plane = (w' * test_data')' + b_vector;
y_predict(plane >= 0, 1) = 1;
y_predict(plane < 0, 1) = -1;

accu = mean(test_label == y_predict);
