% Build baselines
clear
clc
model_root = './models/';
model_names = {'all','bosphorus','ck','bu3dfe'};

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


for model_index = 1:numel(model_names)
    
    for PCA_mode = 0:1
        fprintf('pca mode: %d, model: %s\n', PCA_mode, model_names{model_index});
        
        true_labels = [];
        pred_labels = [];
        accuracy = zeros(1, num_folds);
        % Load models
        if PCA_mode == 0
            model_path = strcat(model_root,model_names{model_index},'_model.mat');
            load(model_path, 'model');
        else
            model_path = strcat(model_root, 'pca_',model_names{model_index},'_model.mat');
            load(model_path,'model');
        end
        
        for i = 1:num_folds
            train_inds = inds(indices ~= i);
            test_inds = inds(indices == i);
            if PCA_mode == 0
                [acc, true_label, pred_label] = baseline_linear_video_BU4DFE_classify(model, bu4dfe_root, samples, test_inds);
            else
                [acc, true_label, pred_label] = baseline_linear_video_BU4DFE_PCA_classify(model, bu4dfe_root, samples, test_inds, PC, means_norm, stds_norm);
            end
            true_labels = [true_labels; true_label];
            pred_labels = [pred_labels; pred_label];
            accuracy(i) = acc;
            
        end
        save(strcat('./results/','baseline_',model_names{model_index},'_',num2str(PCA_mode),'_result.mat'), 'true_labels', 'pred_labels', 'accuracy');
    end
end