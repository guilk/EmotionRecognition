% run cross validation and grid search on training data, test on testing data
clear
clc

addpath('../models/libsvm/matlab');
addpath('./utilities/');
addpath('./scripts_BU3DFE');
addpath('./scripts_ck');
addpath('./scripts_bosphorus');

pca_file = '../pca/generic_face_rigid.mat';
load(pca_file);


ck_root = '../data/CK+/data';
ck_samples = dir(ck_root);
ck_samples = ck_samples(3:end);
rng default
inds = randperm(numel(ck_samples));
splits = 100;
tr_ck_inds = inds(1:splits);
ts_ck_inds = inds(splits+1:end);

ck_tr_samples = ck_samples(tr_ck_inds);
ck_ts_samples = ck_samples(ts_ck_inds);

bu3dfe_root = '../data/BU_3DFE/data';
info_samples = dir([bu3dfe_root '/*.hog']);
bu3dfe_samples = {};
rng default
for i = 1:numel(info_samples)
    bu3dfe_samples{end+1} = info_samples(i).name;
end
inds = randperm(numel(bu3dfe_samples));
splits = 80;
tr_bu3dfe_inds = inds(1:splits);
ts_bu3dfe_inds = inds(splits+1:end);
bu3dfe_tr_samples = bu3dfe_samples(tr_bu3dfe_inds);
bu3dfe_ts_samples = bu3dfe_samples(ts_bu3dfe_inds);

bosphorus_root = '../data/Bosphorus/data';
info_samples = dir([bosphorus_root '/*.hog']);
bosphorus_samples = {};
rng default
for i = 1:numel(info_samples)
    bosphorus_samples{end+1} = info_samples(i).name;
end
inds = randperm(numel(bosphorus_samples));
splits = 80;
tr_bosphorus_inds = inds(1:splits);
ts_bosphorus_inds = inds(splits+1:end);
bosphorus_tr_samples = bosphorus_samples(tr_bosphorus_inds);
bosphorus_ts_samples = bosphorus_samples(ts_bosphorus_inds);

num_folds = 10;
c = 10.^(-6:0.5:1);
e = 10.^(-3);

[accuracy,best_settings] = validate_grid_search(ck_root, ck_tr_samples, bu3dfe_root, bu3dfe_tr_samples,...
    bosphorus_root, bosphorus_tr_samples, c, e, num_folds);

c = best_settings.c;
% c = 0.1;

[tr_ck_features, tr_ck_labels] = prepare_CK_data(ck_root, ck_samples, tr_ck_inds);
[tr_bu3dfe_features, tr_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, tr_bu3dfe_inds);
[tr_bosphorus_features, tr_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, tr_bosphorus_inds);

[ts_ck_features, ts_ck_labels] = prepare_CK_data(ck_root, ck_samples, ts_ck_inds);
size(ts_ck_features,1)
[ts_bu3dfe_features, ts_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, ts_bu3dfe_inds);
size(ts_bu3dfe_features,1)
[ts_bosphorus_features, ts_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, ts_bosphorus_inds);
size(ts_bosphorus_features,1)

tr_features = [tr_ck_features;tr_bu3dfe_features;tr_bosphorus_features];
tr_labels = [tr_ck_labels; tr_bu3dfe_labels; tr_bosphorus_labels];
ts_features = [ts_ck_features; ts_bu3dfe_features; ts_bosphorus_features];
ts_labels = [ts_ck_labels; ts_bu3dfe_labels; ts_bosphorus_labels];

% % Temporary
% no_train_ne_ind = find(tr_labels ~= 0);
% tr_features = tr_features(no_train_ne_ind,:);
% tr_labels = tr_labels(no_train_ne_ind);
% 
% no_test_ne_ind = find(ts_labels ~= 0);
% ts_features = ts_features(no_test_ne_ind,:);
% ts_labels = ts_labels(no_test_ne_ind);

% csvwrite('tr_features.dat',tr_features);
% csvwrite('tr_labels.dat',tr_labels);
% csvwrite('ts_features.dat',ts_features);
% csvwrite('ts_labels.dat',ts_labels);

tr_features = double(sparse(tr_features));
tr_labels = double(tr_labels);
ts_features = double(sparse(ts_features));
ts_labels = double(ts_labels);




clabel = unique(tr_labels);
opt = ['-c ' num2str(c) ' -t 0 -b 1 -q'];
model = svmtrain(tr_labels, tr_features, opt);
[predictions,overall_acc,probs] = svmpredict(ts_labels, ts_features, model, '-b 1');
acc = zeros(length(clabel),1 );

for i = 1:length(clabel)
    c = clabel(i);
    idx = find(ts_labels == c);
    curr_pred_label = predictions(idx);
    curr_gnd_label = ts_labels(idx);
    acc(i) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
end
average_acc = mean(acc);
fprintf('mean accuracy: %f\n', mean(acc));
save three_image_model.mat model





