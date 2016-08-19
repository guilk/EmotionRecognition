% Scripts for ck+ dataset on BU4DFE dataset
clear
clc

% Add utilities path
addpath('./plots');
addpath('./retrieve_BU4DFE/');
addpath('../models/libsvm/matlab');
addpath('./utilities');
addpath('../pca');

bu4dfe_root = '../data/BU_4DFE/';
folders = dir(bu4dfe_root);
folders = folders(3:end);
option = 'hog';
samples = {};
for i = 1:numel(folders)
    samples{end+1} = folders(i).name;
end

% Load generic PCA
pca_file = 'generic_face_rigid.mat';
load(pca_file);

% Train/Test splits
rng default
num_folds = 2;
indices = crossvalind('Kfold', length(samples), num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);

% main function
true_labels = [];
pred_labels = [];
for PCA_mode = 0:1
    [aug_tr_features, aug_tr_labels, model] = load_pretrained_data('ck', PCA_mode); 
    for featAug_mode = 0:1
        for i = 1:num_folds
            fprintf('%dth cross validation\n',i);
            train_inds = inds(indices ~= i);
            test_inds = inds(indices == i);
            if PCA_mode == 0
                [tr_features, tr_labels] = prepare_BU4DFE_data(bu4dfe_root, samples, train_inds, model);
                if featAug_mode == 1
                    tr_features = [tr_features; aug_tr_features];
                    tr_labels = [tr_labels; aug_tr_labels];
                end
                [acc, true_label, pred_label] = linear_video_BU4DFE_classify(tr_features, tr_labels, bu4dfe_root, samples, test_inds);
            else
                [tr_features, tr_labels] = prepare_BU4DFE_PCA_data(bu4dfe_root, samples, train_inds, model, PC, means_norm, stds_norm);
                if featAug_mode == 1
                    tr_features = [tr_features; aug_tr_features];
                    tr_labels = [tr_labels; aug_tr_labels];
                end
                [acc, true_label, pred_label] = linear_video_BU4DFE_PCA_classify(tr_features, tr_labels, bu4dfe_root, samples, test_inds, PC, means_norm, stds_norm);
                
            end
        end
    end
end
    
    
    
