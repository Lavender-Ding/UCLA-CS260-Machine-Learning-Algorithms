function [b,w,entropy] = grad_descent(T, Data, Label, step)

b = 0.1;
w = zeros(size(Data,1),1);
entropy = zeros(1,T);

for i = 1 : T
    sigmoid = 1 ./ (1 + exp(-(b + w' * Data)));
    w = w - step * ((sigmoid - Label) * Data')';
    b = b - step * sum(sigmoid - Label);
    sigmoid = 1 ./ (1 + exp(-(b + w' * Data)));
    sigmoid = max(sigmoid, 1e-16);
    sigmoid = min(sigmoid, 1 - 1e-16);
%     Label_1 = Label(Label(1,:) == 1);
%     sigmoid_1 = sigmoid(Label(1,:) == 1);
%     Label_0 = Label(Label(1,:) == 0);
%     sigmoid_0 = sigmoid(Label(1,:) == 0);
    entropy(1,i) = - sum(Label .* log(sigmoid) + (1 - Label) .* log(1 - sigmoid));
end

end