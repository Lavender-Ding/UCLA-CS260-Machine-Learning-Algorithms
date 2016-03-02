function [encode_result]=encoding(fid)
% fid=fopen('car_train.data');
c=textscan(fid,'%[^,\n],%[^,\n],%[^,\n],%[^,\n],%[^,\n],%[^,\n],%[^,\n]');
Data=[c{1,1},c{1,2},c{1,3},c{1,4},c{1,5},c{1,6},c{1,7}];
f_size=[size(unique(Data(:,1)),1),size(unique(Data(:,2)),1),...
    size(unique(Data(:,3)),1),size(unique(Data(:,4)),1),...
    size(unique(Data(:,5)),1),size(unique(Data(:,6)),1),size(unique(Data(:,7)),1)];
buying=zeros(size(Data,1),f_size(1,1));
buying(strcmp(Data(:,1),'low'),1)=1;
buying(strcmp(Data(:,1),'med'),2)=1;
buying(strcmp(Data(:,1),'high'),3)=1;
buying(strcmp(Data(:,1),'vhigh'),4)=1;
maint=zeros(size(Data,1),f_size(1,2));
maint(strcmp(Data(:,2),'low'),1)=1;
maint(strcmp(Data(:,2),'med'),2)=1;
maint(strcmp(Data(:,2),'high'),3)=1;
maint(strcmp(Data(:,2),'vhigh'),4)=1;
doors=zeros(size(Data,1),f_size(1,3));
doors(strcmp(Data(:,3),'2'),1)=1;
doors(strcmp(Data(:,3),'3'),2)=1;
doors(strcmp(Data(:,3),'4'),3)=1;
doors(strcmp(Data(:,3),'5more'),4)=1;
persons=zeros(size(Data,1),f_size(1,4));
persons(strcmp(Data(:,4),'2'),1)=1;
persons(strcmp(Data(:,4),'4'),2)=1;
persons(strcmp(Data(:,4),'more'),3)=1;
lug_boot=zeros(size(Data,1),f_size(1,5));
lug_boot(strcmp(Data(:,5),'small'),1)=1;
lug_boot(strcmp(Data(:,5),'med'),2)=1;
lug_boot(strcmp(Data(:,5),'big'),3)=1;
safety=zeros(size(Data,1),f_size(1,6));
safety(strcmp(Data(:,6),'low'),1)=1;
safety(strcmp(Data(:,6),'med'),2)=1;
safety(strcmp(Data(:,6),'high'),3)=1;
class(strcmp(Data(:,7),'unacc'),1)=1;
class(strcmp(Data(:,7),'acc'),1)=2;
class(strcmp(Data(:,7),'good'),1)=3;
class(strcmp(Data(:,7),'vgood'),1)=4;
encode_result=[buying,maint,doors,persons,lug_boot,safety,class];
end



