clear;
clc;

fid=fopen('car_train.data');
result=encoding(fid);
train_data=result(:,1:size(result,2)-1);
train_label=result(:,size(result,2));
fid=fopen('car_valid.data');
result=encoding(fid);
new_data=result(:,1:size(result,2)-1);
new_label=result(:,size(result,2));

[valid_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label);


fprintf('Using Naive Bayes: \n');
fprintf('the train accuracy is %8.4f\n',train_accu);
fprintf('the valid accuracy is %8.4f\n',valid_accu);

fid=fopen('car_test.data');
result=encoding(fid);
new_data=result(:,1:size(result,2)-1);
new_label=result(:,size(result,2));

[test_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label);

fprintf('the test accuracy is %8.4f\n',test_accu);



