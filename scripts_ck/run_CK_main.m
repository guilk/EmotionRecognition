%%run_seperate_main
clear
clc

addpath('../../models/libsvm/matlab');
addpath('../utilities/');

prepare_model = 1;
pca_file = '../../pca/generic_face_rigid.mat';
load(pca_file);

feat_root = '../../data/CK+/data';
sub_folders = dir(feat_root);
sub_folders = sub_folders(3:end);
if prepare_model == 1
    inds = 1:length(sub_folders);
    [tr_features, tr_labels] = prepare_CK_data(feat_root, sub_folders, inds);
    no_train_ne_inds = find(tr_labels ~= 0);
    tr_features = tr_features(no_train_ne_inds,:);
    tr_labels = tr_labels(no_train_ne_inds,:);
    
    csvwrite('tr_features.dat',tr_features);
    csvwrite('tr_labels.dat',tr_labels);
    c = 0.1;
    opt = ['-c ' num2str(c) ' -t 0 -b 1 -q'];
    tr_features = double(sparse(tr_features));
    tr_labels = double(tr_labels);
    model = svmtrain(tr_labels, tr_features, opt);
    save ck_model.mat model
end



rng default

num_folds = 10;
indices = crossvalind('Kfold',length(sub_folders),num_folds);
inds = 1:length(sub_folders);
accuracy = zeros(1,num_folds);
for i = 1:num_folds
    fprintf('%dth cross validation\n',i);
    train_inds = inds(indices ~= i);
    test_inds = inds(indices == i);
    [tr_features,tr_labels] = prepare_CK_data(feat_root,sub_folders, train_inds);
    [ts_features,ts_labels] = prepare_CK_data(feat_root,sub_folders, test_inds);
    tr_features = get_pca(tr_features, PC, means_norm, stds_norm);
    ts_features = get_pca(ts_features, PC, means_norm, stds_norm);
    
    [overall_acc, average_acc, probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels);
    accuracy(i) = overall_acc(1);
end
fprintf('Mean accuracy of %d folds cross validation: %f\n',num_folds, mean(accuracy));
%