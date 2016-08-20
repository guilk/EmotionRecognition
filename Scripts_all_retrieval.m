% Scripts for all three image datasets on BU4DFE 
% Scripts for ck+ dataset on BU4DFE dataset
clear
clc
% parpool('local',10);
% Add utilities path
addpath('./plots');
addpath('./retrieve_BU4DFE/');
addpath('../models/libsvm/matlab');
addpath('./utilities');
addpath('../pca');

name_dataset = 'all';
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
num_folds = 10;
indices = crossvalind('Kfold', length(samples), num_folds);
inds = 1:length(samples);
% main function
c = 0.031;
for PCA_mode = 0
    [aug_tr_features, aug_tr_labels, model] = load_pretrained_data(name_dataset, PCA_mode);
    for featAug_mode = 0:1
        
        true_labels = [];
        pred_labels = [];
        accuracy = zeros(1,num_folds);
        for i = 1:num_folds
            fprintf('pca mode: %d, feature augment mode: %d, %dth cross validation\n',PCA_mode, featAug_mode, i);
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
            true_labels = [true_labels; true_label];
            pred_labels = [pred_labels; pred_label];
            accuracy(i) = acc;  
        end
        save(strcat('./results/',num2str(c),'_',name_dataset,'_',num2str(PCA_mode),'_',num2str(featAug_mode),'_result.mat'), 'true_labels', 'pred_labels', 'accuracy');        
    end
end



