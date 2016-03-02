function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CS260 2015 Fall, Homework 2

t_size=size(train_data,1);
n_size=size(new_data,1);

p_c1=length(train_label(train_label==1))/t_size;
p_c2=length(train_label(train_label==2))/t_size;
p_c3=length(train_label(train_label==3))/t_size;
p_c4=length(train_label(train_label==4))/t_size;

%p_c=[p_c1,p_c2,p_c3,p_c4];

conditional_p_c1=sum(train_data(train_label==1,:))./size(train_data(train_label==1,:),1);
conditional_p_c2=sum(train_data(train_label==2,:))./size(train_data(train_label==2,:),1);
conditional_p_c3=sum(train_data(train_label==3,:))./size(train_data(train_label==3,:),1);
conditional_p_c4=sum(train_data(train_label==4,:))./size(train_data(train_label==4,:),1);
conditional_p_c1(conditional_p_c1==0)=0.01;
conditional_p_c2(conditional_p_c2==0)=0.01;
conditional_p_c3(conditional_p_c3==0)=0.01;
conditional_p_c4(conditional_p_c4==0)=0.01;

predict_t_label=zeros(t_size,1);
predict_n_label=zeros(n_size,1);

for i=1:t_size
    one_data=train_data(i,:);
    one_data_p1_1=one_data.*conditional_p_c1;
    one_data_p1_2=(1-one_data).*(1-conditional_p_c1);
    p1=sum(log(one_data_p1_1(one_data~=0)))+...
        sum(log(one_data_p1_2((1-one_data)~=0)))+log(p_c1);
    one_data_p2_1=one_data.*conditional_p_c2;
    one_data_p2_2=(1-one_data).*(1-conditional_p_c2);
    p2=sum(log(one_data_p2_1(one_data~=0)))+...
        sum(log(one_data_p2_2((1-one_data)~=0)))+log(p_c2);
    one_data_p3_1=one_data.*conditional_p_c3;
    one_data_p3_2=(1-one_data).*(1-conditional_p_c3);
    p3=sum(log(one_data_p3_1(one_data~=0)))+...
        sum(log(one_data_p2_2((1-one_data)~=0)))+log(p_c3);
    one_data_p4_1=one_data.*conditional_p_c4;
    one_data_p4_2=(1-one_data).*(1-conditional_p_c4);
    p4=sum(log(one_data_p4_1(one_data~=0)))+...
        sum(log(one_data_p4_2((1-one_data)~=0)))+log(p_c4);
    p_c=[p1,p2,p3,p4];
    [val,pos]=max(p_c);
    predict_t_label(i,1)=pos;
end

for i=1:n_size
    one_data=new_data(i,:);
    one_data_p1_1=one_data.*conditional_p_c1;
    one_data_p1_2=(1-one_data).*(1-conditional_p_c1);
    p1=sum(log(one_data_p1_1(one_data~=0)))+...
        sum(log(one_data_p1_2((1-one_data)~=0)))+log(p_c1);
    one_data_p2_1=one_data.*conditional_p_c2;
    one_data_p2_2=(1-one_data).*(1-conditional_p_c2);
    p2=sum(log(one_data_p2_1(one_data~=0)))+...
        sum(log(one_data_p2_2((1-one_data)~=0)))+log(p_c2);
    one_data_p3_1=one_data.*conditional_p_c3;
    one_data_p3_2=(1-one_data).*(1-conditional_p_c3);
    p3=sum(log(one_data_p3_1(one_data~=0)))+...
        sum(log(one_data_p2_2((1-one_data)~=0)))+log(p_c3);
    one_data_p4_1=one_data.*conditional_p_c4;
    one_data_p4_2=(1-one_data).*(1-conditional_p_c4);
    p4=sum(log(one_data_p4_1(one_data~=0)))+...
        sum(log(one_data_p4_2((1-one_data)~=0)))+log(p_c4);
    p_c=[p1,p2,p3,p4];
    [val,pos]=max(p_c);
    predict_n_label(i,1)=pos;
end

train_accu=mean(predict_t_label==train_label);
new_accu=mean(predict_n_label==new_label);



end