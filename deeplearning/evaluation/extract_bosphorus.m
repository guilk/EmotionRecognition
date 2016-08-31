function extract_bosphorus()

disp('Extract CNN features');
model_root = '/usr0/home/liangkeg/InMind/FG/models/caffe/models/';
% model_def_file = 'bvlc_alexnet/deploy.prototxt';
% model_file = 'bvlc_alexnet/bvlc_alexnet.caffemodel';

model_def_file = 'vgg_face_caffe/VGG_FACE_deploy.prototxt';
model_file = 'vgg_face_caffe/VGG_FACE.caffemodel';

% model_def_file = 'vgg/VGG_ILSVRC_16_layers_deploy.prototxt';
% model_file = 'vgg/VGG_ILSVRC_16_layers.caffemodel';

def_path = fullfile(model_root,model_def_file);
model_path = fullfile(model_root, model_file);

net = caffe.Net(def_path, model_path,'test');
caffe.set_mode_gpu();
gpu_id = 1;
caffe.set_device(gpu_id);


dataset_root = '/usr0/home/liangkeg/InMind/FG/data_cnn/Bosphorus';
dst_root = '/usr0/home/liangkeg/InMind/FG/features/Bosphorus';
folders = dir(dataset_root);
folders = folders(3:end);

for i = 1:numel(folders)
    folder = folders(i).name;
    folder_path = fullfile(dataset_root, folder);
    dst_folder_path = fullfile(dst_root, folder);
    if ~exist(dst_folder_path,'dir')
        mkdir(dst_folder_path);
    end
    images = dir([folder_path '/*.png']);
    im_list = {};
    for j = 1:numel(images)
        im_list{end+1} = fullfile(folder_path,images(j).name);
    end
    extract_features_vgg_face(net, im_list, dst_folder_path);
end

caffe.reset_all();
end
