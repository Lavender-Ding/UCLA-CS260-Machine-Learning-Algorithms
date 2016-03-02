% function data = readall_txt(path)
%   READALL_TXT    读取所有文件
%   DATA = READALL_TXT(PATH)读取路径PATH下的所有txt文件中的数据赋给data
%   txt文件中含有一个数据项
%   输出cell格式以免各txt中数据长度不同
%
%   原始版本：V1.0   作者：李鹏   时间：2009.04.04

function feature = train_feature(Dic,path)

A = dir(fullfile(path,'*.txt'));
% 读取后A的格式为      
%                   name  -- filename
%                   date  -- modification date
%                   bytes -- number of bytes allocated to the file
%                   isdir -- 1 if name is a directory and 0 if not
A = struct2cell(A);
num = size(A);
for k =1:num(2)
    x(k) = A(1,k);% 找出name序列
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