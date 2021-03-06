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
% Better performance without normalization
normalize = 0;
rng default
indices = crossvalind('Kfold', length(samples),num_folds);
inds = 1:length(samples);
accuracy = zeros(1,num_folds);

true_labels = [];
pred_labels = [];
for i = 1:num_folds
    fprintf('%dth cross validation\n',i);
    train_inds = inds(indices ~= i);
    test_inds = inds(indices == i);
    [tr_features, tr_labels] = get_features_labels(feat_root, samples, train_inds);
    [ts_features, ts_labels] = get_features_labels(feat_root, samples, test_inds);
    
    if(normalize == 1)
        tr_features = normr(tr_features);
        ts_features = normr(ts_features);
    end
    
    tr_features = sparse(double(tr_features));
    tr_labels = double(tr_labels);
    ts_features = sparse(double(ts_features));
    ts_labels = double(ts_labels);
    
    opt = [' -c ' num2str(0.1) ' -t 0 -b 1 -q'];
    model = svmtrain(tr_labels, tr_features, opt);
    [pred_label,overall_acc,~] = svmpredict(ts_labels, ts_features, model, '-b 1');
    true_labels = [true_labels; ts_labels];
    pred_labels = [pred_labels; pred_label];
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));
save('vggface_result.mat', 'true_labels', 'pred_labels','accuracy');
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
            %             feature = feats.fc7;
            %             img_feature = feature(:,5);
            features = [features; img_feature'];
        end
        
    end
end
end
