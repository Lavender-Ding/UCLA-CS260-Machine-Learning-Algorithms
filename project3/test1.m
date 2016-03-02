clear;
clc;

fid_dic=fopen('.\spam\dic.dat');
C_dic=textscan(fid_dic,'%[^,\n]');
Dic=C_dic{1,1};

path_train_spam = '.\spam\train\spam\';
path_train_ham = '.\spam\train\ham\';

feature_spam_train = train_feature(Dic, path_train_spam);
feature_ham_train = train_feature(Dic, path_train_ham);

path_test_spam = '.\spam\test\spam\';
path_test_ham = '.\spam\test\ham\';

feature_spam_test = train_feature(Dic, path_test_spam);
feature_ham_test = train_feature(Dic, path_test_ham);

%%%%%%%%%%%%%%%%% find the most prequent three words%%%%%%%%%%%%%%%%%%%%%%%
num_words = sum(feature_spam_train,2) + sum(feature_ham_train,2);
[val, pos] = sort(num_words);
num_words = size(Dic,1);
word1 = Dic(pos(num_words),1);
word2 = Dic(pos(num_words-1),1);
word3 = Dic(pos(num_words-2),1);
fprintf('{(%s: %d of occurrences), (%s: %d of occurrences), (%s: %d of occurrences)}\n',...
    word1{1,1},val(num_words,1),word2{1,1},val(num_words-1,1),word3{1,1},val(num_words-2,1));

%%%%%%%%%%%%%%%%% find the most prequent three words%%%%%%%%%%%%%%%%%%%%%%%

load('.\ionosphere\iono_train_data.mat');
load('.\ionosphere\iono_train_label.mat');
iono_train_data = ionospheretrain';

label_tmp = b;
iono_train_label = zeros(size(iono_train_data,2),1);
iono_train_label(strcmp(label_tmp(:,1),'b'),1)=1;
iono_train_label(strcmp(label_tmp(:,1),'g'),1)=0;
iono_train_label = iono_train_label';

load('.\ionosphere\iono_test_data.mat');
load('.\ionosphere\iono_test_label.mat');
iono_test_data = ionospheretest';
label_tmp1 = g;
iono_test_label = zeros(size(iono_test_data,2),1);
iono_test_label(strcmp(label_tmp1(:,1),'b'),1)=1;
iono_test_label(strcmp(label_tmp1(:,1),'g'),1)=0;
iono_test_label = iono_test_label';

% for i = 1 : size(iono_train_data,1)
%     if iono_train_data(i,:) == ones(1, size(iono_train_data,2)) * iono_train_data(i,1)
%         iono_train_data(i,:) = [];
%         iono_test_data(i,:) = [];
%         i = i - 1;
%     end
%     if i == size(iono_train_data,1)
%         break;
%     end
% end

spam_train_data = [feature_spam_train, feature_ham_train];
spam_train_label = [ones(1,size(feature_spam_train,2)),zeros(1,size(feature_ham_train,2))];
spam_test_data = [feature_spam_test, feature_ham_test];
spam_test_label = [ones(1,size(feature_spam_test,2)),zeros(1,size(feature_ham_test,2))];

% for i = 1 : size(spam_train_data,1)
%     if spam_train_data(i,:) == ones(1, size(spam_train_data,2)) * spam_train_data(i,1)
%         spam_train_data(i,:) = [];
%         spam_test_data(i,:) = [];
%         i = i - 1;
%     end
% end

Marker = '+o*xsd^v><ph';
Color = [0.6,0,0;
         0.3,0.6,0.9;
         0.6,0.3,0.7;
         0.9,0.6,0.3;
         0.3,0.7,0.6;
         0,0.3,0.7;
         0,0,0.9;
         0,0.8,0;
         0.5,0.5,0.5
         0.3,0.3,0.3;
         0.2,0.8,0.8];

T = 50;
i = 0;
q7_norm_iono_train = zeros(size(0 : 0.05 : 0.5));
figure;
for lamda = 0 : 0.05 : 0.5
    i = i + 1;
    [b_iono,w_iono,entropy_iono_train] = newton_regular(T, iono_train_data, iono_train_label,lamda);
    q7_norm_iono_train(1,i) = sqrt(sum(w_iono.^2));
    plot(1:1:T,entropy_iono_train,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',10);
    hold on;
end
legend('{\it\lambda} = 0','{\it\lambda} = 0.05','{\it\lambda} = 0.1','{\it\lambda} = 0.15','{\it\lambda} = 0.2',...
    '{\it\lambda} = 0.25','{\it\lambda} = 0.3','{\it\lambda} = 0.35','{\it\lambda} = 0.4','{\it\lambda} = 0.45','{\it\lambda} = 0.5');
title('Q7:Cross-entropy on Ionosphere using Newton\primes Method with Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)