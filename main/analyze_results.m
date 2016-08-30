% analyse results
addpath('./plots');
[C,order] = plot_confusion_matrix(true_labels, pred_labels);
fprintf('overall accuracy: %f\n',length(find(true_labels == pred_labels))/length(true_labels));

fprintf('Mean accuracy of %d folds cross validation: %f\n', 10, mean(accuracy));