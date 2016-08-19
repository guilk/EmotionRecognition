function [accuracy, true_labels, pred_labels] = baseline_linear_video_BU4DFE_PCA_classify(model, feat_path, src_samples, inds, PC, means_norm, stds_norm)
%LINEAR_VIDEO_BU4DFE_CLASSIFY Summary of this function goes here
%   Detailed explanation goes here

samples = src_samples(inds);
option = 'hog';
true_labels = [];
pred_labels = [];
count = 0;

for i = 1:numel(samples)
    samples_path = strcat(feat_path, '/', samples{i},'/*.hog');
    sub_samples = dir(samples_path);
    for j = 1:numel(sub_samples)
        % TO DO
        count = count + 1;
        sample_path = strcat(feat_path,'/',samples{i});
        ts_features = load_features(sample_path, sub_samples(j).name, option);
        ts_features = get_pca(ts_features, PC, means_norm, stds_norm);
        
        sample_name = sub_samples(j).name;
        splits = strsplit(sample_name, '.');
        video_label = get_label(splits(1));
        true_labels = [true_labels;video_label];
        ts_labels = zeros(size(ts_features,1),1);
        
        ts_features = double(sparse(ts_features));
        ts_labels = double(ts_labels);
        
        [predictions] = svmpredict(ts_labels, ts_features, model, '-b 1');
        pred_label = mode(predictions);
        pred_labels = [pred_labels;pred_label];
    end
end

accuracy = length(find(true_labels == pred_labels))/length(true_labels);
fprintf('Video-wise prediction: %d of %d samples are correctly classified, accuracy is: %f\n',length(find(true_labels == pred_labels)), length(true_labels), accuracy);
end