% run Bosphorus data
clear
clc

addpath('../../models/libsvm/matlab');
addpath('../utilities/');
pca_file = '../../pca/generic_face_rigid.mat';
load(pca_file);
bosphorus_root = '../../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
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
    [tr_features,tr_labels] = prepare_Bosphorus_data(bosphorus_root, samples, train_inds);
    [ts_features, ts_labels] = prepare_Bosphorus_data(bosphorus_root, samples, test_inds);
    
    tr_features = get_pca(tr_features, PC, means_norm, stds_norm);
    ts_features = get_pca(ts_features, PC, means_norm, stds_norm);
    
    [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));


% inds = randperm(numel(info_samples));
% splits = 80;
% train_inds = inds(1:splits);
% test_inds = inds(splits+1:end);
%
% [tr_features, tr_labels] = prepare_Bosphorus_data(bosphorus_root, samples, train_inds);
% [ts_features, ts_labels] = prepare_Bosphorus_data(bosphorus_root, samples, test_inds);
%
% [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);



