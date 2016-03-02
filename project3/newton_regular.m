function [b, w1, entropy] = newton_regular(T, Data, Label, lamda)

[b,w,en] = grad_descent_regular(5, Data, Label, 0.01, 0.05);
w = [b; w];
Data = [ones(1, size(Data, 2)); Data];
D = size(Data, 1);
H = zeros(D, D);
entropy = zeros(1,T);

for i = 1 : T
    sigmoid = 1 ./ (1 + exp(- w' * Data));
    c = sigmoid .* (1 - sigmoid);
    s_matrix = repmat(c,D,1);
    H = (Data .* s_matrix) * (Data');
    I = 2 * lamda * eye(D,D);
    I(1,1) = 0;
    H = H + I;
    w = w - (pinv(H)) * ((sigmoid - Label) * Data' + lamda * 2 * w')';
    w1 = w(2 : size(w,1),1);
    b = w(1,1);
    sigmoid = 1 ./ (1 + exp(- w' * Data));
    sigmoid = max(sigmoid, 1e-16);
    sigmoid = min(sigmoid, 1 - 1e-16);
    entropy(1,i) = - sum(Label .* log(sigmoid) + ...
        (1 - Label) .* log(1 - sigmoid));
end

end