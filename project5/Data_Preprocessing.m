function [train_out, test_out, tMean, tDeviation] = Data_Preprocessing(train_in,test_in)

Mean = mean(train_in);
Deviation = std(train_in, 0, 1);

tMean = mean(test_in);
tDeviation = std(test_in, 0, 1);

Mean_matrix_train = repmat(Mean, size(train_in, 1), 1);
Deviation_matrix_train = repmat(Deviation, size(train_in, 1), 1);

Mean_matrix_test = repmat(Mean, size(test_in, 1), 1);
Deviation_matrix_test = repmat(Deviation, size(test_in, 1), 1);

train_out = (train_in - Mean_matrix_train) ./ Deviation_matrix_train;
test_out = (test_in - Mean_matrix_test) ./ Deviation_matrix_test;

end