% Test seperate image dataset on video datasets
clear
clc

addpath('../models/liblinear-2.1/matlab');

% feat_root = '../data/CK+/data';
% sub_folders = dir(feat_root);
% sub_folders = sub_folders(3:end);
% 
% inds = 1:length(sub_folders);
% [tr_features, tr_labels] = prepare_CK_data(feat_root,sub_folders, inds);
% 
% no_train_ne_ind = find(tr_labels ~= 0);
% tr_features = tr_features(no_train_ne_ind,:);
% tr_labels = tr_labels(no_train_ne_ind);
% tr_features = double(sparse(tr_features));
% tr_labels = double(tr_labels);
% 
% opt = [' -B 1 -c ' num2str(0.1) ' -s 1 -q'];
% model = train(tr_labels, tr_features, opt);
% 
% save ck_image_model.mat model

% bu3dfe_root = 'E:/2016Spring/InMind/data/Emotion/BU_3DFE/data';
% info_samples = dir([bu3dfe_root '/*.hog']);
% samples = {};
% rng default
% for i = 1:numel(info_samples)
%     samples{end+1} = info_samples(i).name;
% end
% inds = 1:length(samples);
% [tr_features, tr_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, inds);
% no_train_ne_ind = find(tr_labels ~= 0);
% tr_features = tr_features(no_train_ne_ind,:);
% tr_labels = tr_labels(no_train_ne_ind);
% tr_features = double(sparse(tr_features));
% tr_labels = double(tr_labels);
% 
% opt = [' -B 1 -c ' num2str(0.1) ' -s 1 -q'];
% model = train(tr_labels, tr_features, opt);
% save bu3dfe_image_model.mat model

addpath('../models/liblinear-2.1/matlab');
bosphorus_root = '../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
samples = {};

rng default
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end
inds = 1:length(samples);
[tr_features, tr_labels] = prepare_Bosphorus_data(bosphorus_root, samples, inds);
no_train_ne_ind = find(tr_labels ~= 0);
tr_features = tr_features(no_train_ne_ind,:);
tr_labels = tr_labels(no_train_ne_ind);
tr_features = double(sparse(tr_features));
tr_labels = double(tr_labels);

opt = [' -B 1 -c ' num2str(0.1) ' -s 1 -q'];
model = train(tr_labels, tr_features, opt);
save bosphorus_image_model.mat model

