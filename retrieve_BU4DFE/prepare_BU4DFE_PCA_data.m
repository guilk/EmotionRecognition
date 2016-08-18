function [dst_features, dst_labels] = prepare_BU4DFE_PCA_data(feat_path, src_samples, inds, model, PC, means_norm, stds_norm)

samples = src_samples(inds);
option = 'hog';
dst_features = [];
dst_labels = [];

for i = 1:numel(samples)
    samples_path = strcat(feat_path, '/', samples{i},'/*.hog');
    sub_samples = dir(samples_path);
    for j = 1:numel(sub_samples)
        sample_path = strcat(feat_path, '/', samples{i});
        features = load_features(sample_path, sub_samples(j).name, option);        
        features = get_pca(features, PC, means_norm, stds_norm);
        fake_labels = zeros(size(features,1),1);
        
        ori_features = double(sparse(features));
        fake_labels = double(fake_labels);
        [~, ~, probs] = svmpredict(fake_labels, ori_features,model, '-b 1');
        
        sample_name = sub_samples(j).name;
        splits = strsplit(sample_name,'.');
        label = get_label(splits(1));
        
        selected_probs = probs(:,label);
        index = find(selected_probs == max(selected_probs));
        tr_feature = features(index,:);
        tr_label = label * ones(size(tr_feature,1),1);
        
        dst_features = [dst_features; tr_feature];
        dst_labels = [dst_labels; tr_label];
    end
end

end