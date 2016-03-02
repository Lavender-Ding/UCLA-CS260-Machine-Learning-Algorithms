% function data = readall_txt(path)
%   READALL_TXT    ��ȡ�����ļ�
%   DATA = READALL_TXT(PATH)��ȡ·��PATH�µ�����txt�ļ��е����ݸ���data
%   txt�ļ��к���һ��������
%   ���cell��ʽ�����txt�����ݳ��Ȳ�ͬ
%
%   ԭʼ�汾��V1.0   ���ߣ�����   ʱ�䣺2009.04.04

function feature = train_feature(Dic,path)

A = dir(fullfile(path,'*.txt'));
% ��ȡ��A�ĸ�ʽΪ      
%                   name  -- filename
%                   date  -- modification date
%                   bytes -- number of bytes allocated to the file
%                   isdir -- 1 if name is a directory and 0 if not
A = struct2cell(A);
num = size(A);
for k =1:num(2)
    x(k) = A(1,k);% �ҳ�name����
end
numDic = size(Dic,1);
feature = zeros(numDic,k);

for k = 1:num(2)
    newpath = strcat(path,'\',x(k));
    FID = fopen(char(newpath));
    C = textscan(FID,'%s');
    Data = C{1,1};
    for i=1:numDic
        feature(i,k) = sum(strcmp(Data(:,1),Dic(i,1)));
    end
    fclose(FID);
end

end