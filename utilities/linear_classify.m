function [overall_acc, average_acc,probs] = linear_classify(tr_features, tr_labels, ts_features, ts_labels)
tr_features = double(sparse(tr_features));
tr_labels = double(tr_labels);
ts_features = double(sparse(ts_features));
ts_labels = double(ts_labels);

clabel = unique(ts_labels);
opt = [' -c ' num2str(0.1) ' -t 0 -b 1 -q'];
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
end