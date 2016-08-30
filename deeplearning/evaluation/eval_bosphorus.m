function eval_bosphorus()

dataset_root = '/usr0/home/liangkeg/InMind/FG/data_cnn/Bosphorus';
dst_root = '/usr0/home/liangkeg/InMind/FG/features/Bosphorus';
folders = dir(dataset_root);
folders = folders(3:end);

for i = 1:numel(folders)
    folder = folders(i).name;
    folder_path = fullfile(dataset_root, folder);
    images = dir([folder_path '/*.png']);
    im_list = {};
    for j = 1:numel(images)
        im_list{end+1} = fullfile(folder_path,images(j).name);
    end
    extract_features(im_list, dst_root);
end
end