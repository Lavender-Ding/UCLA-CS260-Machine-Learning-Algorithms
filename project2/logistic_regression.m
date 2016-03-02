clear;
clc;

fid=fopen('car_train.data');
result=encoding(fid);
train_data=result(:,1:size(result,2)-1);
train_label=result(:,size(result,2));
fid=fopen('car_valid.data');
result=encoding(fid);
valid_data=result(:,1:size(result,2)-1);
valid_label=result(:,size(result,2));
fid=fopen('car_test.data');
result=encoding(fid);
test_data=result(:,1:size(result,2)-1);
test_label=result(:,size(result,2));

Factor = mnrfit(train_data, train_label); 
trainScores = mnrval(Factor, train_data); 
[~,train_label_logistic]=max(trainScores,[],2);
train_accu_logistic=mean(train_label_logistic==train_label);

validScores = mnrval(Factor, valid_data); 
[~,valid_label_logistic]=max(validScores,[],2);
valid_accu_logistic=mean(valid_label_logistic==valid_label);

testScores = mnrval(Factor, test_data); 
[~,test_label_logistic]=max(testScores,[],2);
test_accu_logistic=mean(test_label_logistic==test_label);

fprintf('Using Logistic Regression: \n');
fprintf('the train accuracy is %8.4f\n',train_accu_logistic);
fprintf('the valid accuracy is %8.4f\n',valid_accu_logistic);
fprintf('the test accuracy is %8.4f\n',test_accu_logistic);





