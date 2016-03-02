clear;
clc;

Train = load('splice_train.mat');
TrainData_in = Train.data;
TrainLabel_all = Train.label;
Test = load('splice_test.mat');
TestData_in = Test.data;
TestLabel = Test.label;

[TrainData_all, TestData, tMean, tDeviation] = Data_Preprocessing(TrainData_in, TestData_in);

N_train = size(TrainData_all, 1);
D = size(TrainData_all, 2);

for j = 1 : 5
    ValidData = TrainData_all((j - 1) * N_train / 5 + 1 : j * N_train / 5, :);
    ValidLabel = TrainLabel_all((j - 1) * N_train / 5 + 1 : j * N_train / 5, :);
    tmpData = TrainData_all;
    tmpLabel = TrainLabel_all;
    tmpData((j - 1) * N_train / 5 + 1 : j * N_train / 5, :) = [];
    tmpLabel((j - 1) * N_train / 5 + 1 : j * N_train / 5, :) = [];
    TrainData = tmpData;
    TrainLabel = tmpLabel;
    for i = 1 : 9
        C = 4 ^ (i - 7);
        C_vector(1, i) = C;
        tic
        [w, b] = trainsvm(TrainData, TrainLabel, C);
        t = toc;
        ValidAccu_all(j, i) = testsvm(ValidData, ValidLabel, w, b);
        TrainTime_all(j, i) = t;
    end
end

ValidAccu_average = mean(ValidAccu_all);
TrainTime = mean(TrainTime_all);

fprintf('Average validation accuracy by my own implementation of SVM\n');
fprintf('C                        %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector);
fprintf('Cross-valiation accuracy %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', ValidAccu_average);
fprintf('Average training time by my own implementation of SVM\n');
fprintf('C             %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector);
fprintf('Training time %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', TrainTime);

[Val, Pos] = max(ValidAccu_average);
C = 4 ^ (Pos - 7);
[w, b] = trainsvm(TrainData_all, TrainLabel_all, C);
TestAccu = testsvm(TestData, TestLabel, w, b);

%% Linear LibSVM

for i = 1 : 9
    C = 4 ^ (i - 7);
    C_vector_linear(1, i) = C;
    tmp = num2str(C);
    tic
    LinearModel(1, i) = svmtrain(TrainLabel_all,TrainData_all, ['-c ' tmp ' -t 0 -v 5 -q']);
    t = toc;
    LinearModelTime(1, i) = t / 5;
end

fprintf('Average validation accuracy using linear LibSVM\n');
fprintf('C                        %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector_linear);
fprintf('Cross-valiation accuracy %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', LinearModel);
fprintf('Average training time using linear LibSVM\n');
fprintf('C             %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector_linear);
fprintf('Training time %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', LinearModelTime);
% [TestLabelPredict, TestAccu, PredictValue] =svmpredict(TestLabel, TestData, model);

%% Polynomial kernel LibSVM

for d = 1 : 3
    for i = 1 : 11
        C = 4 ^ (i - 4);
        C_vector_poly(1, i) = C;
        tmp = num2str(C);
        tmp_d = num2str(d);
        tic
        PolyModel(d, i) = svmtrain(TrainLabel_all,TrainData_all, ['-c ' tmp ' -t 1 -d ' tmp_d ' -v 5 -q']);
        t = toc;
        PolyModelTime(d, i) = t / 5;
    end
end

fprintf('Average validation accuracy by LibSVM using Polynomial kernel\n');
fprintf('C              %8.4f%8.4f%8.4f   %d  %d  %d   %d   %d     %d     %d   %d\n', C_vector_poly);
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 1, PolyModel(1, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 2, PolyModel(2, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 3, PolyModel(3, :));

fprintf('Average training time by LibSVM using Polynomial kernel\n');
fprintf('C              %8.4f%8.4f%8.4f   %d  %d  %d   %d   %d     %d     %d     %d\n', C_vector_poly);
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 1, PolyModelTime(1, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 2, PolyModelTime(2, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 3, PolyModelTime(3, :));

%% RBF kernel LibSVM

for g = 1 : 6
    gamma = 4 ^ (g - 8);
    g_vector(1, g) = gamma;
    tmp_g = num2str(gamma);
    for i = 1 : 11
        C = 4 ^ (i - 4);
        C_vector_rbf(1, i) = C;
        tmp = num2str(C);
        tic
        RBFModel(g, i) = svmtrain(TrainLabel_all,TrainData_all, ['-c ' tmp ' -t 2 -g ' tmp_g ' -v 5 -q']);
        t = toc;
        RBFModelTime(g, i) = t / 5;
    end
end

[x y]=find(RBFModel==max(max(RBFModel)));
C = 4 ^ (y - 4);
tmp_C = num2str(C);
gamma = 4 ^ (x - 8);
tmp_g = num2str(gamma);
test_model = svmtrain(TrainLabel_all,TrainData_all, ['-c ' tmp ' -t 2 -g ' tmp_g ' -q']);
[predict_label, TestAccu_kernel, dec_values] =svmpredict(TestLabel, TestData, test_model);

%% output
fprintf('\n\nThe mean and the standard deviation of the third feature on the test data are: %8.4f %8.4f\n',tMean(3),tDeviation(3));
fprintf('The mean and the standard deviation of the tenth feature on the test data are: %8.4f %8.4f\n',tMean(10),tDeviation(10));

fprintf('Average validation accuracy by my own implementation of SVM\n');
fprintf('C                        %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector);
fprintf('Cross-valiation accuracy %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', ValidAccu_average .* 100);
fprintf('Average training time by my own implementation of SVM\n');
fprintf('C             %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector);
fprintf('Training time %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', TrainTime);

fprintf('Test accuracy using the C which has the highest accuracy on cross validation is: %8.1f\n', TestAccu * 100);

fprintf('Average validation accuracy using linear LibSVM\n');
fprintf('C                        %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector_linear);
fprintf('Cross-valiation accuracy %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', LinearModel);
fprintf('Average training time using linear LibSVM\n');
fprintf('C             %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', C_vector_linear);
fprintf('Training time %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', LinearModelTime);

fprintf('Average validation accuracy by LibSVM using Polynomial kernel\n');
fprintf('C              %8.4f%8.4f%8.4f   %d  %d  %d   %d   %d     %d     %d   %d\n', C_vector_poly);
fprintf('Degree =%d    %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', 1, PolyModel(1, :));
fprintf('Degree =%d    %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', 2, PolyModel(2, :));
fprintf('Degree =%d    %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', 3, PolyModel(3, :));

fprintf('Average training time by LibSVM using Polynomial kernel\n');
fprintf('C              %8.4f%8.4f%8.4f   %d  %d  %d   %d   %d     %d     %d     %d\n', C_vector_poly);
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 1, PolyModelTime(1, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 2, PolyModelTime(2, :));
fprintf('Degree =%d    %8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n', 3, PolyModelTime(3, :));

fprintf('Average validation accuracy by LibSVM using Polynomial kernel\n');
fprintf('C                   %8.4f%8.4f%8.4f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', C_vector_rbf);
for g = 1 : 6
    fprintf('gamma =%8.4f    %8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', g_vector(1, g), RBFModel(g, :));
end

fprintf('Average training time by LibSVM using Polynomial kernel\n');
fprintf('C                   %8.4f%8.4f%8.4f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n', C_vector_rbf);
for g = 1 : 6
    fprintf('gamma =%8.4f    %8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n', g_vector(1, g), RBFModelTime(g, :));
end

fprintf('Test accuracy using kernel and parameters which maximize cross validation accuracy is: %8.1f\n', TestAccu_kernel(1,1));










