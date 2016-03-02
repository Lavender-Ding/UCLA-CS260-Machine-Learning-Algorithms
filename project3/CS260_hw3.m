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

%%%%%%%%%%%%%%%%Gradient Descent without Regularization%%%%%%%%%%%%%%%%%%%%
T = 50;
i = 0;
figure;
q3_norm_spam = zeros(1,5);
for step = [0.001,0.01,0.05,0.1,0.5]
    i = i + 1;
    [b_spam,w_spam,entropy_spam] = grad_descent(T, spam_train_data, spam_train_label, step);
    q3_norm_spam(1,i) = sqrt(sum(w_spam.^2));
    plot(1:1:T,entropy_spam,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',11);
    hold on;
end
legend('{\it\eta} = 0.001','{\it\eta} = 0.01','{\it\eta} = 0.05','{\it\eta} = 0.1','{\it\eta} = 0.5');
title('Q3a:Cross-entropy on EmailSpam using Gradient Descent without Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

i = 0;
figure;
q3_norm_iono = zeros(1,5);
for step = [0.001,0.01,0.05,0.1,0.5]
    i = i + 1;
    [b_iono,w_iono,entropy_iono] = grad_descent(T, iono_train_data, iono_train_label, step);
    q3_norm_iono(1,i) = sqrt(sum(w_iono.^2));
    plot(1:1:T,entropy_iono,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',11);
    hold on;
end
legend('{\it\eta} = 0.001','{\it\eta} = 0.01','{\it\eta} = 0.05','{\it\eta} = 0.1','{\it\eta} = 0.5');
title('Q3a:Cross-entropy on Ionosphere using Gradient Descent without Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

fprintf('L2 norm(without regularization) %8.4f%8.4f%8.4f%8.4f%8.4f\n',[0.001,0.01,0.05,0.1,0.5]);
fprintf('Ionosphere                      %8.4f%8.4f%8.4f%8.4f%8.3f\n',q3_norm_iono);
fprintf('EmailSpam                       %8.4f%8.4f%8.4f%8.4f%8.3f\n',q3_norm_spam);

%%%%%%%%%%%%%%%%Gradient Descent without Regularization%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%Gradient Descent with Regularization%%%%%%%%%%%%%%%%%%%%%

i = 0;
figure;
for step = [0.001,0.01,0.05,0.1,0.5]
    i = i + 1;
    [b_spam,w_spam,entropy_spam] = grad_descent_regular(T, spam_train_data, spam_train_label, step, 0.1);
    plot(1:1:T,entropy_spam,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',11);
    hold on;
end
legend('{\it\eta} = 0.001','{\it\eta} = 0.01','{\it\eta} = 0.05','{\it\eta} = 0.1','{\it\eta} = 0.5');
title('Q4a:Cross-entropy on EmailSpam using Gradient Descent with Regularization({\it\lambda=0.1})','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

i = 0;
figure;
for step = [0.001,0.01,0.05,0.1,0.5]
    i = i + 1;
    [b_iono,w_iono,entropy_iono] = grad_descent_regular(T, iono_train_data, iono_train_label, step, 0.1);
    plot(1:1:T,entropy_iono,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',11);
    hold on;
end
legend('{\it\eta} = 0.001','{\it\eta} = 0.01','{\it\eta} = 0.05','{\it\eta} = 0.1','{\it\eta} = 0.5');
title('Q4a:Cross-entropy on Ionosphere using Gradient Descent with Regularization({\it\lambda=0.1})','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

q4_norm_spam = zeros(1,11);
q4_norm_iono = zeros(1,11);
i = 0;
for lamda = 0 : 0.05 : 0.5
    i = i + 1;
    [b_spam,w_spam,entropy_spam] = grad_descent_regular(T, spam_train_data, spam_train_label, 0.01, lamda);
    [b_iono,w_iono,entropy_iono] = grad_descent_regular(T, iono_train_data, iono_train_label, 0.01, lamda);
    q4_norm_spam(1,i) = sqrt(sum(w_spam.^2));
    q4_norm_iono(1,i) = sqrt(sum(w_iono.^2));
end
fprintf('L2 norm(with regularization,0.01) %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n',0 : 0.05 : 0.5);
fprintf('Ionosphere                        %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n',q4_norm_iono);
fprintf('EmailSpam                         %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n',q4_norm_spam);

for step = [0.001,0.01,0.05,0.1,0.5]
    entropy_spamtrain_50 = zeros(size(0 : 0.05 : 0.5));
    entropy_ionotrain_50 = zeros(size(0 : 0.05 : 0.5));
    entropy_spamtest = zeros(size(0 : 0.05 : 0.5));
    entropy_ionotest = zeros(size(0 : 0.05 : 0.5));
    for i = 1 : 11
        [b_spam,w_spam,entropy_spam_train] = grad_descent_regular(T, spam_train_data, spam_train_label, step, (i - 1) * 0.05);
        [b_iono,w_iono,entropy_iono_train] = grad_descent_regular(T, iono_train_data, iono_train_label, step, (i - 1) * 0.05);
%         entropy_spamtrain_50(1,i) = entropy_spam_train(1,50);
%         entropy_ionotrain_50(1,i) = entropy_iono_train(1,50);
        entropy_spamtrain_50(1,i) = compute_entropy(b_spam, w_spam, spam_train_data, spam_train_label);
        entropy_ionotrain_50(1,i) = compute_entropy(b_iono, w_iono, iono_train_data, iono_train_label);
        entropy_spamtest(1,i) = compute_entropy(b_spam, w_spam, spam_test_data, spam_test_label);
        entropy_ionotest(1,i) = compute_entropy(b_iono, w_iono, iono_test_data, iono_test_label);
    end
    figure;
    plot(0 : 0.05 : 0.5,entropy_ionotrain_50,'LineWidth',2,'Color',Color(1,:),'Marker',Marker(1),'MarkerSize',11);
    hold on;
    plot(0 : 0.05 : 0.5,entropy_ionotest,'LineWidth',2,'Color',Color(2,:),'Marker',Marker(2),'MarkerSize',11);
    legend('trainging data','test data');
    title(['Q4c:Cross-entropy on Ionosphere at {\itT}=50 for different regularization cofficients,{\it\eta} =',num2str(step)],'FontName','Times New Roman','FontWeight','Bold','FontSize',16);
    xlabel('{\it\lambda}','FontName','Times New Roman','FontSize',14) 
    ylabel('Cross-entropy at {\itT}=50','FontName','Times New Roman','FontSize',14,'Rotation',90)
    figure;
    plot(0 : 0.05 : 0.5,entropy_spamtrain_50,'LineWidth',2,'Color',Color(1,:),'Marker',Marker(1),'MarkerSize',11);
    hold on;
    plot(0 : 0.05 : 0.5,entropy_spamtest,'LineWidth',2,'Color',Color(2,:),'Marker',Marker(2),'MarkerSize',11);
    legend('trainging data','test data');
    title(['Q4c:Cross-entropy on EmailSpam at {\itT}=50 for different regularization cofficients,{\it\eta} =',num2str(step)],'FontName','Times New Roman','FontWeight','Bold','FontSize',16);
    xlabel('{\it\lambda}','FontName','Times New Roman','FontSize',14) 
    ylabel('Cross-entropy at {\itT}=50','FontName','Times New Roman','FontSize',14,'Rotation',90)

end

%%%%%%%%%%%%%%%%%%Gradient Descent with Regularization%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculate Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [b_spam,w_spam,entropy_spam_train] = grad_descent_regular(T, spam_train_data, spam_train_label, 0.1, 0.1);
% prediction_spam = b_spam + w_spam' * spam_test_data;
% prediction_spam(prediction_spam(1,:) >= 0.5) = 1;
% prediction_spam(prediction_spam(1,:) < 0.5) = 0;
% spam_accu=mean(prediction_spam==spam_test_label);
% 
% [b_iono,w_iono,entropy_iono_train] = grad_descent_regular(T, iono_train_data, iono_train_label, 0.1, 0.1);
% prediction_iono = b_iono + w_iono' * iono_test_data;
% prediction_iono(prediction_iono(1,:) >= 0.5) = 1;
% prediction_iono(prediction_iono(1,:) < 0.5) = 0;
% iono_accu=mean(prediction_iono==iono_test_label);
%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculate Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%Newton without Regularization%%%%%%%%%%%%%%%%%%%%%%%%
T = 50;
[b_spam,w_spam,entropy_spam_train] = newton(T, spam_train_data, spam_train_label);
q6_norm_spam_train = sqrt(sum(w_spam.^2));
[b_iono,w_iono,entropy_iono_train] = newton(T, iono_train_data, iono_train_label);
q6_norm_iono_train = sqrt(sum(w_iono.^2));
figure;
plot(1:1:T,entropy_spam_train,'LineWidth',2,'Color',Color(1,:),'Marker',Marker(1),'MarkerSize',11);
title('Q6a:Cross-entropy on EmailSpam using Newton\primes Method without Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)
figure;
plot(1:1:T,entropy_iono_train,'LineWidth',2,'Color',Color(2,:),'Marker',Marker(2),'MarkerSize',11);
title('Q6a:Cross-entropy on Ionosphere using Newton\primes Method without Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

fprintf('L2 norm for training data(Newton method):\n');
fprintf('Ionosphere: %8.4f\n',q6_norm_iono_train);
fprintf('EmailSpam:  %8.4f\n',q6_norm_spam_train);

q6_norm_spam_test = compute_entropy(b_spam, w_spam, spam_test_data, spam_test_label);
q6_norm_iono_test = compute_entropy(b_iono, w_iono, iono_test_data, iono_test_label);

fprintf('Entropy for testing data(Newton method):\n');
fprintf('Ionosphere: %8.4f\n',q6_norm_iono_test);
fprintf('EmailSpam:  %8.4f\n',q6_norm_spam_test);

% %%%%%%%%%%%%%%%%%%%%%%Newton without Regularization%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%Newton with Regularization%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
i = 0;
q7_norm_spam_train = zeros(size(0 : 0.05 : 0.5));
q7_norm_spam_test = zeros(size(0 : 0.05 : 0.5));;
figure;
for lamda = 0 : 0.05 : 0.5
    i = i + 1;
    [b_spam,w_spam,entropy_spam_train] = newton_regular(T, spam_train_data, spam_train_label,lamda);
    q7_norm_spam_train(1,i) = sqrt(sum(w_spam.^2));
    q7_norm_spam_test(1,i) = compute_entropy(b_spam, w_spam, spam_test_data, spam_test_label);
    plot(1:1:T,entropy_spam_train,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',10);
    hold on;
end
legend('{\it\lambda} = 0','{\it\lambda} = 0.05','{\it\lambda} = 0.1','{\it\lambda} = 0.15','{\it\lambda} = 0.2',...
    '{\it\lambda} = 0.25','{\it\lambda} = 0.3','{\it\lambda} = 0.35','{\it\lambda} = 0.4','{\it\lambda} = 0.45','{\it\lambda} = 0.5');
title('Q7:Cross-entropy on EmailSpam using Newton\primes Method with Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

i = 0;
q7_norm_iono_train = zeros(size(0 : 0.05 : 0.5));
q7_norm_iono_test = zeros(size(0 : 0.05 : 0.5));
figure;
for lamda = 0 : 0.05 : 0.5
    i = i + 1;
    [b_iono,w_iono,entropy_iono_train] = newton_regular(T, iono_train_data, iono_train_label,lamda);
    q7_norm_iono_train(1,i) = sqrt(sum(w_iono.^2));
    q7_norm_iono_test(1,i) = compute_entropy(b_iono, w_iono, iono_test_data, iono_test_label);
    plot(1:1:T,entropy_iono_train,'LineWidth',2,'Color',Color(i,:),'Marker',Marker(i),'MarkerSize',10);
    hold on;
end
legend('{\it\lambda} = 0','{\it\lambda} = 0.05','{\it\lambda} = 0.1','{\it\lambda} = 0.15','{\it\lambda} = 0.2',...
    '{\it\lambda} = 0.25','{\it\lambda} = 0.3','{\it\lambda} = 0.35','{\it\lambda} = 0.4','{\it\lambda} = 0.45','{\it\lambda} = 0.5');
title('Q7:Cross-entropy on Ionosphere using Newton\primes Method with Regularization','FontName','Times New Roman','FontWeight','Bold','FontSize',16);
xlabel('Number of iterations','FontName','Times New Roman','FontSize',14) 
ylabel('Cross-entropy','FontName','Times New Roman','FontSize',14,'Rotation',90)

fprintf('L2 norm for training data(Regularized Newton method):\n');
fprintf('lambda:     %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n',0 : 0.05 : 0.5);
fprintf('Ionosphere: %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n',q7_norm_iono_train);
fprintf('EmailSpam:  %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n',q7_norm_spam_train);

fprintf('entropy for testing data(Newton method):\n');
fprintf('lambda:     %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n',0 : 0.05 : 0.5);
fprintf('Ionosphere: %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n',q7_norm_iono_test);
fprintf('EmailSpam:  %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n',q7_norm_spam_test);

%%%%%%%%%%%%%%%%%%%%%%%Newton with Regularization%%%%%%%%%%%%%%%%%%%%%%%%%%












