function [accuracy, true_labels, pred_labels] = linear_video_BU4DFE_classify(tr_features, tr_labels, feat_path, src_samples, inds)
%LINEAR_VIDEO_BU4DFE_CLASSIFY Summary of this function goes here
%   Detailed explanation goes here
tr_features = double(sparse(tr_features));
tr_labels = double(tr_labels);
opt = ['-c ' num2str(0.1) ' -t 0 -b 1 -q'];
model = svmtrain(tr_labels, tr_features, opt);

samples = src_samples(inds);
option = 'hog';
true_labels = [];
pred_labels = [];
count = 0;
% linear classify sample by sample
% load bosphorus_image_model.mat model

for i = 1:numel(samples)
    samples_path = strcat(feat_path, '/', samples{i},'/*.hog');
    sub_samples = dir(samples_path);
    for j = 1:numel(sub_samples)
        % TO DO
        count = count + 1;
        sample_path = strcat(feat_path,'/',samples{i});
        ts_features = load_features(sample_path, sub_samples(j).name, option);
        sample_name = sub_samples(j).name;
        splits = strsplit(sample_name, '.');
        video_label = get_label(splits(1));
        true_labels = [true_labels;video_label];
        ts_labels = zeros(size(ts_features,1),1);
        
        ts_features = double(sparse(ts_features));
        ts_labels = double(ts_labels);
        
        [predictions] = svmpredict(ts_labels, ts_features, model);
        pred_label = mode(predictions);
        pred_labels = [pred_labels;pred_label];
    end
end

accuracy = length(find(true_labels == pred_labels))/length(true_labels);
fprintf('Video-wise prediction: %d of %d samples are correctly classified, accuracy is: %f\n',length(find(true_labels == pred_labels)), length(true_labels), accuracy);
end

