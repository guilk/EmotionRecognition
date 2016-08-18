% run cross validation and grid search on training data, test on testing data
clear
clc

addpath('../models/liblinear-2.1/matlab');
num_folds = 10;
ck_root = '../data/CK+/data';
ck_samples = dir(ck_root);
ck_samples = ck_samples(3:end);
rng default
ck_indices = crossvalind('Kfold',length(ck_samples),num_folds);
ck_inds = 1:length(ck_samples);

bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/data';
info_samples = dir([bu3dfe_root '/*.hog']);
bu3dfe_samples = {};
rng default
for i = 1:numel(info_samples)
    bu3dfe_samples{end+1} = info_samples(i).name;
end
bu3dfe_indices = crossvalind('Kfold',length(bu3dfe_samples),num_folds);
bu3dfe_inds = 1:length(bu3dfe_samples);

bosphorus_root = '../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
bosphorus_samples = {};
rng default
for i = 1:numel(info_samples)
    bosphorus_samples{end+1} = info_samples(i).name;
end
bosphorus_indices = crossvalind('Kfold',length(bosphorus_samples),num_folds);
bosphorus_inds = 1:length(bosphorus_samples);

c = 0.1;
accuracy = zeros(1,num_folds);
for i = 1:num_folds
    tr_ck_inds = ck_inds(ck_indices ~= i);
    ts_ck_inds = ck_inds(ck_indices == i);
    tr_bu3dfe_inds = bu3dfe_inds(bu3dfe_indices ~= i);
    ts_bu3dfe_inds = bu3dfe_inds(bu3dfe_indices == i);
    tr_bosphorus_inds = bosphorus_inds(bosphorus_indices ~= i);
    ts_bosphorus_inds = bosphorus_inds(bosphorus_indices == i);
    [tr_ck_features, tr_ck_labels] = prepare_CK_data(ck_root, ck_samples, tr_ck_inds);
    [tr_bu3dfe_features, tr_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, tr_bu3dfe_inds);
    [tr_bosphorus_features, tr_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, tr_bosphorus_inds);
    
    [ts_ck_features, ts_ck_labels] = prepare_CK_data(ck_root, ck_samples, ts_ck_inds);
    size(ts_ck_features,1)
    [ts_bu3dfe_features, ts_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, ts_bu3dfe_inds);
    size(ts_bu3dfe_features,1)
    [ts_bosphorus_features, ts_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, ts_bosphorus_inds);
    size(ts_bosphorus_features,1)
    
    tr_features = [tr_ck_features;tr_bu3dfe_features;tr_bosphorus_features];
    tr_labels = [tr_ck_labels; tr_bu3dfe_labels; tr_bosphorus_labels];
    ts_features = [ts_ck_features; ts_bu3dfe_features; ts_bosphorus_features];
    ts_labels = [ts_ck_labels; ts_bu3dfe_labels; ts_bosphorus_labels];
    
    [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));

