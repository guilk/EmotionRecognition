function [dst_features, dst_labels] = prepare_Bosphorus_data(feat_path, src_samples, inds)
option = 'hog';
samples = src_samples(inds);

dst_features = [];
dst_labels = [];

for i = 1:numel(samples)
    feature = load_features(feat_path,samples{i},option);
    splits = strsplit(samples{i},'.');
    label_path = fullfile(feat_path,[splits{1} '_info_label.txt']);
    labels = csvread(label_path);
    inds = find(labels >= 0);
    dst_label = labels(inds);
    dst_feature = feature(inds,:);
    dst_features = [dst_features;dst_feature];
    dst_labels = [dst_labels;dst_label];
end

end