% Scripts for ck+ dataset on BU4DFE dataset
clear
clc

% Add utilities path
addpath('./plots');
addpath('./retrieve_BU4DFE/');
addpath('../models/libsvm/matlab');
addpath('./utilities');

bu4dfe_root = '../data/BU_4DFE/';
folders = dir(bu3dfe_root);
folders = folders(3:end);
option = 'hog';
for i = 1:numel(folders)
    samples{end+1} = folders{i}.name;
end

aug_tr_features = csvread('./models/ck_tr_features.dat');
aug_tr_labels = csvread('./models/ck_tr_labels.dat');

