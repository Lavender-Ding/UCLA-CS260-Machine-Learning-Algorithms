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

train_accu_tree_gdi=zeros(1,10);
valid_accu_tree_gdi=zeros(1,10);
valid_accu_tree_gdi=zeros(1,10);
for i=1:10
    tree=fitctree(train_data,train_label,'MinLeafSize',i,...
        'Prune','off','SplitCriterion','gdi');
    train_label_tree=predict(tree, train_data);
    train_accu_tree_gdi(1,i)=mean(train_label_tree==train_label);
    valid_label_tree=predict(tree, valid_data);
    valid_accu_tree_gdi(1,i)=mean(valid_label_tree==valid_label);
    test_label_tree=predict(tree, test_data);
    test_accu_tree_gdi(1,i)=mean(test_label_tree==test_label);
end

train_accu_tree_cro=zeros(1,10);
valid_accu_tree_cro=zeros(1,10);
valid_accu_tree_cro=zeros(1,10);
for i=1:10
    tree=fitctree(train_data,train_label,'MinLeafSize',i,...
        'Prune','off','SplitCriterion','deviance');
    train_label_tree=predict(tree, train_data);
    train_accu_tree_cro(1,i)=mean(train_label_tree==train_label);
    valid_label_tree=predict(tree, valid_data);
    valid_accu_tree_cro(1,i)=mean(valid_label_tree==valid_label);
    test_label_tree=predict(tree, test_data);
    test_accu_tree_cro(1,i)=mean(test_label_tree==test_label);
end

fprintf('Using Decision Tree: \n');
fprintf('For i=1 to 10, using Gini index, train accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',train_accu_tree_gdi);
fprintf('For i=1 to 10, using Gini index, valid accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',valid_accu_tree_gdi);
fprintf('For i=1 to 10, using Gini index, test accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',test_accu_tree_gdi);
fprintf('For i=1 to 10, using Cross entropy, train accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',train_accu_tree_cro);
fprintf('For i=1 to 10, using Cross entropy, valid accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',valid_accu_tree_cro);
fprintf('For i=1 to 10, using Cross entropy, test accuracy is\n%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n',test_accu_tree_cro);









