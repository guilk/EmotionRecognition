%% select dataset for evaluation
% Test ck+, Bosphorus, BU_3DFE and BU_4DFE dataset to select one for final
% evaluation
function select_dataset_for_evaluation()
clear;
clc;

addpath('../../models/libsvm/matlab');

% add paths to features
addpath('../../data/Bosphorus/data');
addpath('../../data/BU_3DFE/data');
addpath('../../data/BU_4DFE/');
addpath('../../data/CK+');

% add path to pca
addpath('../../pca/');
addpath('../utilities');
addpath('../plots');

addpath('../scripts_ck');
addpath('../scripts_BU3DFE');
addpath('../scripts_BU4DFE');
addpath('../scripts_bosphorus');

pca_file = 'generic_face_rigid.mat';
load(pca_file);

[ck_features, ck_labels] = get_ck_data();
[bos_features, bos_labels] = get_bosphorus_data();
[bu3dfe_features, bu3dfe_labels] = get_bu3dfe_data();
[bu4dfe_features, bu4dfe_labels] = get_bu4dfe_data();

% case 1: test ck+ dataset, train on the other three datasets
tr_features = [bos_features; bu3dfe_features; bu4dfe_features];
tr_labels = [bos_labels; bu3dfe_labels; bu4dfe_labels];
ts_features = ck_features;
ts_labels = ck_labels;

[C, order] = linear_pca_classify(tr_features, tr_labels, ts_features, ts_labels, PC, means_norm, stds_norm);

% case 2: test bosphorus dataset, train on the other three datasets
tr_features = [ck_features; bu3dfe_features; bu4dfe_features];
tr_labels = [ck_labels; bu3dfe_labels; bu4dfe_labels];
ts_features = bos_features;
ts_labels = bos_labels;

[C, order] = linear_pca_classify(tr_features, tr_labels, ts_features, ts_labels, PC, means_norm, stds_norm);

% case 3: test bu3dfe dataset, train on the other three datasets
tr_features = [ck_features; bos_features; bu4dfe_features];
tr_labels = [ck_labels; bos_labels; bu4dfe_labels];
ts_features = bu3dfe_features;
ts_labels = bu3dfe_labels;

[C, order] = linear_pca_classify(tr_features, tr_labels, ts_features, ts_labels, PC, means_norm, stds_norm);
% case 4: test bu4dfe dataset, train on the other three datasets
tr_features = [ck_features; bos_features; bu3dfe_features];
tr_labels = [ck_labels; bos_labels; bu3dfe_labels];
ts_features = bu4dfe_features;
ts_labels = bu4dfe_labels;

[C, order] = linear_pca_classify(tr_features, tr_labels, ts_features, ts_labels, PC, means_norm, stds_norm);
end

function [C, order] = linear_pca_classify(tr_features, tr_labels, ts_features, ts_labels, PC, means_norm, stds_norm)
tr_features = get_pca(tr_features, PC, means_norm, stds_norm);
ts_features = get_pca(ts_features, PC, means_norm, stds_norm);

tr_features = double(sparse(tr_features));
tr_labels = double(tr_labels);
ts_features = double(sparse(ts_features));
ts_labels = double(ts_labels);
c = 0.001;
opt = ['-c ' num2str(c) ' -t 0 -b 1 -q'];
model = svmtrain(tr_labels, tr_features, opt);
[predictions] = svmpredict(ts_labels, ts_features, model);
[C,order] = plot_confusion_matrix(ts_labels, predictions);
end

function [dst_features,dst_labels] = get_ck_data()
feat_root = '../../data/CK+/data';
sub_folders = dir(feat_root);
sub_folders = sub_folders(3:end);

inds = 1:length(sub_folders);
[dst_features, dst_labels] = prepare_CK_data(feat_root, sub_folders, inds);
end
function [dst_features, dst_labels] = get_bosphorus_data()
bosphorus_root = '../../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
samples = {};
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end
inds = 1:length(samples);
[dst_features, dst_labels] = prepare_Bosphorus_data(bosphorus_root, samples, inds);

end
function [dst_features, dst_labels] = get_bu3dfe_data()
bu3dfe_root = '../../data/BU_3DFE/data';
info_samples = dir([bu3dfe_root '/*.hog']);
samples = {};
for i = 1:numel(info_samples)
    samples{end+1} = info_samples(i).name;
end
inds = 1:length(samples);
[dst_features, dst_labels] = prepare_BU3DFE_data(bu3dfe_root, samples, inds);
end
function [dst_features, dst_labels] = get_bu4dfe_data()
num_neutral = 101;
bu4dfe_root = '../../data/BU_4DFE';
folders = dir(bu4dfe_root);
folders = folders(3:end);
samples = {};
for i = 1:numel(folders)
    samples{end+1} = folders(i).name;
end
inds = 1:length(samples);
[dst_features, dst_labels] = prepare_BU4DFE_data(bu4dfe_root, samples, inds);

neu_index = find(dst_labels == 0);
non_neu_index = find(dst_labels ~= 0);

rng default;
inds = randperm(length(neu_index),num_neutral);
neu_labels = dst_labels(inds);
neu_features = dst_features(inds,:);

non_neu_labels = dst_labels(non_neu_index);
non_neu_features = dst_features(non_neu_index,:);

dst_features = [neu_features;non_neu_features];
dst_labels = [neu_labels; non_neu_labels];
end
