function [b, w1, entropy] = newton(T, Data, Label)

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
%     for j = 1 : D
%         for k = j : D
%             H(j,k) = sum(Data(j,:) .* Data(k,:) .* sigmoid .* (1 - sigmoid));
%             H(k,j) = H(j,k);
%         end
%     end
    w = w - (pinv(H)) * ((sigmoid - Label) * Data')';
    w1 = w(2 : size(w,1),1);
    b = w(1,1);
    sigmoid = 1 ./ (1 + exp(- w' * Data));
    sigmoid = max(sigmoid, 1e-16);
    sigmoid = min(sigmoid, 1 - 1e-16);
    entropy(1,i) = - sum(Label .* log(sigmoid) + (1 - Label) .* log(1 - sigmoid));
end

end