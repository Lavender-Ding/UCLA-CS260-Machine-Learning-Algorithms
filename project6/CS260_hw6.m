clear;
clc;

load('face_data.mat');
num_images = size(image, 2);
reshape_images = zeros(num_images, 2500);
for i = 1 : num_images
    reshape_images(i, :) = reshape(image{1, i}, 1, []);
end

d = 200;
deigenvecs = pca_fun(reshape_images, d);

figure;
for i = 1 : 5
    eigenface = reshape(deigenvecs(:, i), [50, 50]);
    subplot(2, ceil(5/2), i);
    imshow(eigenface, []);
    title(['eigenface ',num2str(i)],'FontName','Times New Roman','FontWeight','Bold','FontSize',14);
end

%% Classification

D = [20, 25, 50, 100, 200];
D1 = [20, 50, 100, 200];
D2 = [25, 100, 200];
fsize = [70, 120, 120, 140, 190];

for di = 1 : size(D, 2)
    d = D(di);
    deigenvecs = pca_fun(reshape_images, d);
    images_pca = reshape_images * deigenvecs;
    for i = 1 : 9
        C = 4 ^ (i - 7);
        C_vector_linear(1, i) = C;
        tmp = num2str(C);
        for f = 1 : 5
            validation = images_pca(subsetID(1,:) == f, :);
            valid_label = personID(:, subsetID(1,:) == f)';
            training = images_pca;
            training(subsetID(1,:) == f, :) = [];
            train_label = personID';
            train_label(subsetID(1,:) == f, :) = [];
            LinearModel = svmtrain(train_label,training, ['-c ' tmp ' -t 0 -q']);
            [predict_label, Accu, dec_values] =svmpredict(valid_label, validation, LinearModel);
            Valid_Accu_all(1, f) = Accu(1, 1);
        end
        Valid_Accu = sum(fsize .* Valid_Accu_all) / sum(fsize);
%         Valid_Accu = mean(Valid_Accu_all);
        Valid_Accu_C(1, i) = Valid_Accu;
    end
    [val, pos] = max(Valid_Accu_C);
    Test_Accu(1, di) = val;
    Optimal_C(1, di) = 4 ^ (pos - 7);
end
C1 = Optimal_C;
C1(:, 2) = [];
fprintf('Optimal C for each d using linear LibSVM\n');
fprintf('d         %d  %d  %d  %d\n', D1);
fprintf('Optimal C %8.3f%8.3f%8.3f%8.3f\n', C1);

% D2 = [25, 100, 200];
% for di = 1 : size(D2, 2)
%     d = D2(di);
%     deigenvecs = pca_fun(reshape_images, d);
%     images_pca = reshape_images * deigenvecs;
%     for f = 1 : 5
%         validation = images_pca(subsetID(1,:) == f, :);
%         valid_label = personID(:, subsetID(1,:) == f)';
%         training = images_pca;
%         training(subsetID(1,:) == f, :) = [];
%         train_label = personID';
%         train_label(subsetID(1,:) == f, :) = [];
%         LinearModel = svmtrain(train_label,training, ['-c 1 -t 0 -q']);
%         [predict_label, Accu, dec_values] =svmpredict(valid_label, validation, LinearModel);
%         Test_Accu_all(1, f) = Accu(1, 1);
%     end
%     Test_Accu(1, di) = sum(fsize .* Test_Accu_all) / sum(fsize);
% %     Test_Accu(1, di) = mean(Test_Accu_all);
% end
% 
% fprintf('Optimal C for each d using linear LibSVM\n');
% fprintf('d         %d  %d  %d  %d\n', D);
% fprintf('Optimal C %8.3f%8.3f%8.3f%8.3f\n\n', Optimal_C);
Test_Accu(:, 1) = [];
Test_Accu(:, 3) = [];
fprintf('Average test accuracy using linear LibSVM\n');
fprintf('d         %d  %d  %d\n', D2);
fprintf('Accuracy %8.2f%8.2f%8.2f%8.2f\n', Test_Accu);









