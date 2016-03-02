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

for k=1:2:23
    [new_accu, train_accu] = ...
        knn_classify(train_data, train_label, new_data, new_label, k);
    validaccuracy(:,(k+1)/2)=[k;train_accu;new_accu];
end

fid=fopen('car_test.data');
result=encoding(fid);
new_data=result(:,1:size(result,2)-1);
new_label=result(:,size(result,2));

for k=1:2:23
    [new_accu, train_accu] = ...
        knn_classify(train_data, train_label, new_data, new_label, k);
    testaccuracy(:,(k+1)/2)=[k;train_accu;new_accu];
end

accuracy=[validaccuracy;testaccuracy(3,:)];

% t_size=size(train_data,1);
% n_size=size(new_data,1);
% 
% newPredictLabel=zeros(n_size,1);
% trainPredictLabel=zeros(t_size,1);
% 
% for i=1:n_size
%     tmp=repmat(new_data(i,:),t_size,1);
%     dismap=(tmp-train_data).^2;
%     distance=sum(dismap,2);
%     [dis,num]=sort(distance);
%     K_near=num(1:k,1);
%     K_near_labels=train_label(K_near,1);
%     labelCount=tabulate(K_near_labels);
%     [maxVal,maxLabel]=max(labelCount(:,2));
%     newPredictLabel(i,1)=labelCount(maxLabel,1);
% end
% 
% for i=1:t_size
%     t_tmp=repmat(train_data(i,:),t_size-1,1);
%     trainout=train_data;
%     trainout(i,:)=[];
%     t_dismap=(t_tmp-trainout).^2;
%     t_distance=sum(t_dismap,2);
%     [t_dis,t_num]=sort(t_distance);
%     t_K_near=t_num(1:k,1);
%     t_K_near_labels=train_label(t_K_near,1);
%     t_labelCount=tabulate(t_K_near_labels);
%     [t_maxVal,t_maxLabel]=max(t_labelCount(:,2));
%     trainPredictLabel(i,1)=t_labelCount(t_maxLabel,1);
% end
% 
% new_accu=mean(newPredictLabel==new_label);
% train_accu=mean(trainPredictLabel==train_label);


% t_size=size(train_data,1);
% n_size=size(new_data,1);
% 
% PredictLabel=zeros(n_size,1);
% 
% for i=1:n_size
%     tmp=repmat(new_data(i,:),t_size,1);
%     dismap=(tmp-train_data).^2;
%     distance=sum(dismap,2);
%     [dis,num]=sort(distance);
%     K_near=num(1:k,1);
%     K_near_labels=train_label(K_near,1);
%     labelCount=tabulate(K_near_labels);
%     [maxVal,maxLabel]=max(labelCount(:,2));
%     PredictLabel(i,1)=labelCount(maxLabel,1);
% end


