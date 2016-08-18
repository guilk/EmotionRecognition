function [dst_features, dst_labels] = prepare_CK_data(feat_path,samples, inds)

% 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

train_samples = get_samples(samples,inds);
[dst_features, dst_labels] = prepare_data(feat_path,train_samples);


end

function [sub_samples] = get_samples(samples, inds)

sub_samples = {};
for i = 1:numel(inds)
    sub_samples{end+1} = samples(inds(i)).name;
end

end

function [features, labels] = prepare_data(feat_root,data_list)

option = 'hog';
file_name = 'hog_features.hog';
label_name = 'label.txt';

features = [];
labels = [];
for i = 1:numel(data_list)
    seq_folders_path = fullfile(feat_root, data_list{i});
    seq_folders = dir(seq_folders_path);
    seq_folders = seq_folders(3:end);
    
    for j = 1:numel(seq_folders)
        feat_path = fullfile(seq_folders_path,seq_folders(j).name);
        feature = load_features(feat_path,file_name,option);
        label = int8(load(fullfile(feat_path,'label.txt')));
        [dst_feature,dst_label] = parse_features(feature,label);
        features = [features;dst_feature];
        labels = [labels; dst_label];
    end
end

end

function [dst_feature,dst_label] = parse_features(src_feature,src_label)

% Treat the first frame as neutral state
dst_feature = src_feature(1,:);
dst_label = [0];
% if src_label = 2, The data is labled as contempt not one of seven basic
% emotions, preserve the first frame skip the rest.
if src_label == 2
    return
end
true_label = 0;
switch src_label
    case 0
        true_label = 0;
    case 1
        true_label = 1;
    otherwise
        true_label = src_label - 1;
end
dst_feature = [dst_feature;src_feature(end,:)];
dst_label = [dst_label;true_label];
end