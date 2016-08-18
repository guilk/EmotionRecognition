%%run_seperate_main
clear
clc

addpath('../models/liblinear-2.1/matlab');

feat_root = '../data/CK+/data';
sub_folders = dir(feat_root);
sub_folders = sub_folders(3:end);
train_inds = 1:length(sub_folders);
[ck_feature,ck_label] = prepare_CK_data(feat_root,sub_folders, train_inds);

bosphorus_root = '../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
samples = {};

rng default
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end
train_inds = 1:length(samples);
[bos_feature,bos_label] = prepare_Bosphorus_data(bosphorus_root, samples, train_inds);


bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/data';
info_samples = dir([bu3dfe_root '/*.hog']);
samples = {};
rng default
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end


num_folds = 10;
indices = crossvalind('Kfold',length(samples),num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);

for i = 1:num_folds
    fprintf('%dth cross validation\n',i);
    train_inds = inds(indices ~= i);
    test_inds = inds(indices == i);
    [tr_feature,tr_label] = prepare_BU3DFE_data(bu3dfe_root, samples, train_inds);
    [ts_features, ts_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, test_inds);
    tr_features = [ck_feature; bos_feature; tr_feature];
    tr_labels = [ck_label; bos_label; tr_label];
    
    [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));
