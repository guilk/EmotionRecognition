function [dst_features, dst_labels] = prepare_BU4DFE_data(feat_path, src_samples, inds)
num_samples = 1;
samples = src_samples(inds);
option = 'hog';
dst_features = [];
dst_labels = [];

for i = 1:numel(samples)
    samples_path = strcat(feat_path,'/',samples{i},'/*.hog');
    sub_samples = dir(samples_path);
%     sub_samples = sub_samples(3:end);
    for j = 1:numel(sub_samples)
        sample_path = strcat(feat_path,'/',samples{i});
        features = load_features(sample_path,sub_samples(j).name, option);
        num_frames = size(features,1);
        inds = floor(num_frames/2 - num_samples/2):floor(num_frames/2+num_samples/2);
        feature = features(inds,:);
        sample_name = sub_samples(j).name;
        splits = strsplit(sample_name,'.');
        label = get_label(splits(1));
        
        dst_features = [dst_features;feature];
        dst_labels = [dst_labels;label*ones(size(inds,2),1)];
        
        dst_features = [dst_features;feature(1,:)];
        dst_labels = [dst_labels; 0];
    end
end


end