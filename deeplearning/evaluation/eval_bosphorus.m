function eval_bosphorus()
clear
clc

addpath('../../../models/libsvm/matlab');
addpath('../../utilities');

feat_root = '/usr0/home/liangkeg/InMind/FG/features/Bosphorus';

subjects = dir(feat_root);
samples = {};

for i = 3:numel(subjects)
    samples{end+1} = subjects(i).name;
end

num_folds = 10;
rng default
indices = crossvalind('Kfold', length(samples),num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);

for i = 1:num_folds
    fprintf('%dth cross validation\n',i);
    train_inds = inds(indices ~= i);
    %     test_inds = inds(indices == i);
    [tr_features, tr_labels] = get_features_labels(feat_root, samples, train_inds);
    size(tr_features)
    size(tr_labels)
    
    
end

end

function [features, labels] = get_features_labels(bosphorus_root, src_samples, inds)
samples = src_samples(inds);
features = [];
labels = [];
for i = 1:numel(samples)
    sample_path = fullfile(bosphorus_root, samples{i});
    images = dir(sample_path);
    images = images(3:end);
    
    for j = 1:numel(images)
        splits = strsplit(images(j).name,'_');
        if strcmp(splits{2},'E') || strcmp(splits{2},'N')
            switch splits{3}
                case 'N'
                    labels = [labels; 0];
                case 'ANGER'
                    labels = [labels; 1];
                case 'DISGUST'
                    labels = [labels; 2];
                case 'FEAR'
                    labels = [labels; 3];
                case 'HAPPY'
                    labels = [labels; 4];
                case 'SADNESS'
                    labels = [labels;5];
                case 'SURPRISE'
                    labels = [labels; 6];
            end
            load(fullfile(bosphorus_root, samples{i}, images(j).name));
            img_feature = mean(feats.fc7, 2);
            features = [features; img_feature'];
        end
        
    end
end
end
