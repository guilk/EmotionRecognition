% Test BU_3DFE seperately
clear
clc

addpath('../../models/liblinear-2.1/matlab');
addpath('../utilities/');
pca_file = '../../pca/generic_face_rigid.mat';
load(pca_file);
prepare_mode = 1;

bu3dfe_root = '../../data/BU_3DFE/data';
% bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/data';
info_samples = dir([bu3dfe_root '/*.hog']);
samples = {};
rng default
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end

if prepare_mode == 1
    inds = 1:length(samples);
    [tr_features, tr_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, inds);
    no_train_ne_inds = find(tr_labels ~= 0);
    tr_features = tr_features(no_train_ne_inds,:);
    tr_labels = tr_labels(no_train_ne_inds,:);
    
    csvwrite('../models/bu3dfe_tr_features.dat',tr_features);
    csvwrite('../models/bu3dfe_tr_labels.dat',tr_labels);
    c = 0.1;
    opt = ['-c ' num2str(c) ' -t 0 -b 1 -q'];
    tr_features = double(sparse(tr_features));
    tr_labels = double(tr_labels);
    model = svmtrain(tr_labels, tr_features, opt);
    save('../models/bu3dfe_model.mat','model');
    
    tr_features = get_pca(tr_features, PC, means_norm, stds_norm);
    csvwrite('../models/pca_bu3dfe_tr_features.dat',tr_features);
    csvwrite('../models/pca_bu3dfe_tr_labels.dat',tr_labels);
    
    c = 0.1;
    opt = ['-c ' num2str(c) ' -t 0 -b 1 -q'];
    tr_features = double(sparse(tr_features));
    tr_labels = double(tr_labels);
    model = svmtrain(tr_labels, tr_features, opt);
    save('../models/pca_bu3dfe_model.mat','model');
end


num_folds = 10;
indices = crossvalind('Kfold',length(samples),num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);

for i = 1:num_folds
    fprintf('%dth cross validation\n',i);
    train_inds = inds(indices ~= i);
    test_inds = inds(indices == i);
    %     bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/3d_data';
    [tr_features,tr_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, train_inds);
    [ts_features, ts_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, test_inds);
    tr_features = get_pca(tr_features, PC, means_norm, stds_norm);
    ts_features = get_pca(ts_features, PC, means_norm, stds_norm);
    %     [tr_features_3d,tr_labels_3d] = prepare_BU3DFE_data(bu3dfe_root, samples, train_inds);
    %     [ts_features_3d, ts_labels_3d] = prepare_BU3DFE_data(bu3dfe_root, samples, test_inds);
    %     bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/data';
    %     [tr_features_2d,tr_labels_2d] = prepare_BU3DFE_data(bu3dfe_root, samples, train_inds);
    %     [ts_features_2d, ts_labels_2d] = prepare_BU3DFE_data(bu3dfe_root, samples, test_inds);
    %     tr_features = [tr_features_3d;tr_features_2d];
    %     ts_features = [ts_features_3d;ts_features_2d];
    %     tr_labels = [tr_labels_3d;tr_labels_2d];
    %     ts_labels = [ts_labels_3d;ts_labels_2d];
    
    %     tr_features = normr(tr_features);
    %     ts_features = normr(ts_features);
    
    [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));

% inds = randperm(numel(samples));
% splits = 80;
% train_inds = inds(1:splits);
% test_inds = inds(splits+1:end);
%
% [tr_features,tr_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, train_inds);
% [ts_features, ts_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, test_inds);
%
% [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);




