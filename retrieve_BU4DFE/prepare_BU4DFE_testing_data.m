% Prepare BU4DFE training data
function [dst_features, dst_labels] = prepare_BU4DFE_testing_data(feat_path,src_samples, inds)
samples = src_samples(inds);
option = 'hog';
dst_features = [];
dst_labels = [];

for i = 1:numel(samples)
    samples_path = strcat(feat_path,'/',samples{i},'/*.hog');
    sub_samples = dir(samples_path);
    for j = 1:numel(sub_samples)
        sample_path = strcat(feat_path,'/',samples{i});
        feature = load_features(sample_path,sub_samples(j).name, option);
        sample_name = sub_samples(j).name;
        splits = strsplit(sample_name,'.');
        label = get_label(splits(1));
        
        dst_features = [dst_features;feature];
        dst_labels = [dst_labels;label*ones(size(feature,1),1)];
        
    end
end
end