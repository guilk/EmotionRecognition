function [grid_accuracy, best_setings] = validate_grid_search(ck_root, ck_samples, bu3dfe_root, bu3dfe_samples,...
    bosphorus_root, bosphorus_samples, c, e, num_folds)

% use linear svm to find optimal parameters
% tr_samples: train samples list
% tr_labels : train labels list

% create indices of ck+ dataset
ck_indices = crossvalind('Kfold', length(ck_samples),num_folds);
ck_inds = 1:length(ck_samples);

% create indices of bu3dfe dataset
bu3dfe_indices = crossvalind('Kfold', length(bu3dfe_samples), num_folds);
bu3dfe_inds = 1:length(bu3dfe_samples);

% create indices of bosphorus dataset
bosphorus_indices = crossvalind('Kfold', length(bosphorus_samples), num_folds);
bosphorus_inds = 1:length(bosphorus_samples);

accuracy = zeros(num_folds, length(c));

for i = 1:num_folds
    fprintf('%dth fold cross validation\n',i);
    tr_ck_inds = ck_inds(ck_indices~=i);
    ts_ck_inds = ck_inds(ck_indices==i);
    tr_bu3dfe_inds = bu3dfe_inds(bu3dfe_indices~=i);
    ts_bu3dfe_inds = bu3dfe_inds(bu3dfe_indices==i);
    tr_bosphorus_inds = bosphorus_inds(bosphorus_indices~=i);
    ts_bosphorus_inds = bosphorus_inds(bosphorus_indices==i);
    
    [tr_ck_features, tr_ck_labels] = prepare_CK_data(ck_root, ck_samples, tr_ck_inds);
    [tr_bu3dfe_features, tr_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, tr_bu3dfe_inds);
    [tr_bosphorus_features, tr_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, tr_bosphorus_inds);
    
    [ts_ck_features, ts_ck_labels] = prepare_CK_data(ck_root, ck_samples, ts_ck_inds);
    [ts_bu3dfe_features, ts_bu3dfe_labels] = prepare_BU3DFE_data(bu3dfe_root, bu3dfe_samples, ts_bu3dfe_inds);
    [ts_bosphorus_features, ts_bosphorus_labels] = prepare_Bosphorus_data(bosphorus_root, bosphorus_samples, ts_bosphorus_inds);
    
    tr_features = [tr_ck_features;tr_bu3dfe_features;tr_bosphorus_features];
    tr_labels = [tr_ck_labels; tr_bu3dfe_labels; tr_bosphorus_labels];
    ts_features = [ts_ck_features; ts_bu3dfe_features; ts_bosphorus_features];
    ts_labels = [ts_ck_labels; ts_bu3dfe_labels; ts_bosphorus_labels];
    
    tr_features = double(sparse(tr_features));
    tr_labels = double(tr_labels);
    ts_features = double(sparse(ts_features));
    ts_labels = double(ts_labels);
    
    for j = 1:length(c)
        opt = [' -c ' num2str(c(j)) ' -e ' num2str(e) ' -t 0 -b 1 -q'];
        model = svmtrain(tr_labels, tr_features, opt);
        [~,overall_acc,~] = svmpredict(ts_labels, ts_features, model, '-b 1');
        fprintf('c: %f, e: %f, accuracy: %f\n',c(j),e,overall_acc(1));
        accuracy(i,j) = overall_acc(1);
    end
    
end

grid_accuracy = mean(accuracy);

[max_acc, max_index] = max(grid_accuracy);
fprintf('The max accuracy is %f when c equals to %f\n', max_acc, c(max_index));
best_setings.c = c(max_index);
end