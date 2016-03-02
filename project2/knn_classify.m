function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CS260 2015 Fall, Homework 1

t_size=size(train_data,1);
n_size=size(new_data,1);

newPredictLabel=zeros(n_size,1);
trainPredictLabel=zeros(t_size,1);

for i=1:n_size
    tmp=repmat(new_data(i,:),t_size,1);
    dismap=(tmp-train_data).^2;
    distance=sum(dismap,2);
    [dis,num]=sort(distance);
    K_near=num(1:k,1);
    K_near_labels=train_label(K_near,1);
    labelCount=tabulate(K_near_labels);
    [maxVal,maxLabel]=max(labelCount(:,2));
    maxLabel=labelCount(labelCount(:,2)==maxVal,1);
    randomLabel=maxLabel(randperm(length(maxLabel)));
    newPredictLabel(i,1)=randomLabel(1,1);
%     newPredictLabel(i,1)=labelCount(maxLabel,1);
end

for i=1:t_size
    tmp=repmat(train_data(i,:),t_size-1,1);
    trainout=train_data;
    trainout(i,:)=[];
    labelout=train_label;
    labelout(i,:)=[];
    dismap=(tmp-trainout).^2;
    distance=sum(dismap,2);
    [dis,num]=sort(distance);
    K_near=num(1:k,1);
    K_near_labels=labelout(K_near,1);
    labelCount=tabulate(K_near_labels);
    [maxVal,maxLabel]=max(labelCount(:,2));
    maxLabel=labelCount(labelCount(:,2)==maxVal,1);
    randomLabel=maxLabel(randperm(length(maxLabel)));
    trainPredictLabel(i,1)=randomLabel(1,1);
%     trainPredictLabel(i,1)=labelCount(maxLabel,1);
end

new_accu=mean(newPredictLabel==new_label);
train_accu=mean(trainPredictLabel==train_label);

end

