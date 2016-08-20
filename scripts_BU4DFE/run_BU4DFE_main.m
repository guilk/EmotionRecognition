% run BU_4DFE dataset

clear
clc
% addpath('./scripts_BU4DFE/');
addpath('../../models/libsvm/matlab');
addpath('../plots');
addpath('../utilities');

% aug_tr_features = csvread('tr_features.dat');
% aug_tr_labels = csvread('tr_labels.dat');
% aug_ts_features = csvread('ts_features.dat');
% aug_ts_labels = csvread('ts_labels.dat');

num_samples = 1;
ifVideo = 1;

bu4dfe_root = '../../data/BU_4DFE/';
folders = dir(bu4dfe_root);
folders = folders(3:end);
samples={};
for i = 1:numel(folders)
    samples{end+1} = folders(i).name;
end

rng default
num_folds = 10;
indices = crossvalind('Kfold', length(samples), num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);


if ifVideo == 0
    for i = 1:num_folds
        fprintf('%dth cross validation\n', i);
        train_inds = inds(indices ~= i);
        test_inds = inds(indices == i);
        [tr_features, tr_labels] = prepare_BU4DFE_training_data(bu4dfe_root, num_samples,samples, train_inds);
        [ts_features, ts_labels] = prepare_BU4DFE_testing_data(bu4dfe_root, ifVideo,samples, test_inds);
        [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
        accuracy(i) = overall_acc(1);
    end
else
    true_labels = [];
    pred_labels = [];
    for i = 1:num_folds
        fprintf('%dth cross validation\n',i);
        train_inds = inds(indices ~= i);
        test_inds = inds(indices == i);
        [tr_features, tr_labels] = prepare_BU4DFE_training_data(bu4dfe_root, num_samples, samples, train_inds);
%         tr_features = [tr_features; aug_tr_features];
%         tr_labels = [tr_labels; aug_tr_labels];
        [acc, true_label, pred_label] = linear_video_BU4DFE_classify(tr_features, tr_labels, bu4dfe_root, samples, test_inds);  
        true_labels = [true_labels; true_label];
        pred_labels = [pred_labels; pred_label];
        accuracy(i) = acc;
    end
    [C,order] = plot_confusion_matrix(true_labels, pred_labels);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n', num_folds, mean(accuracy));
