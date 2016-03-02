function entropy_test=compute_entropy(b, w, Data, Label)

sigmoid = 1 ./ (1 + exp(-(b + w' * Data)));
sigmoid = max(sigmoid, 1e-16);
sigmoid = min(sigmoid, 1 - 1e-16);

entropy_test = - sum(Label .* log(sigmoid) + (1 - Label) .* log(1 - sigmoid));

end