function [features, labels, model] = load_pretrained_data(name_dataset, ifPCA)
if ifPCA == 1
    feature_name = strcat('pca_',name_dataset,'_tr_features.dat');
    label_name = strcat('pca_', name_dataset, '_tr_labels.dat');
else
    feature_name = strcat(name_dataset,'_tr_features.dat');
    label_name = strcat(name_dataset,'_tr_labels.dat');
end
    features = csvread(strcat('./models/',feature_name));
    labels = csvread(strcat('./models/', label_name));
    load(strcat('./models/',name_dataset,'_model.mat'), 'model');
end