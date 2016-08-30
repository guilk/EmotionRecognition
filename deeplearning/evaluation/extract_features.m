function extract_features(im_list, dst_folder)

disp('Extract CNN features');
model_root = '/usr0/home/liangkeg/InMind/FG/models/caffe/models/';
model_def_file = 'bvlc_alexnet/deploy.prototxt';
model_file = 'bvlc_alexnet/bvlc_alexnet.caffemodel';

def_path = fullfile(model_root,model_def_file);
model_path = fullfile(model_root, model_file);

net = caffe.Net(def_path, model_path,'test');
caffe.set_mode_gpu();
gpu_id = 1;
caffe.set_device(gpu_id);

for im_index = 1:numel(im_list)
    feats = [];
    I = imread(im_list{im_index});
    input_data = {prepare_image(I)};
    scores = net.forward(input_data);
    feats.fc7 = net.blobs('fc7').get_data();
    
    [~,fname] = fileparts(im_list{im_index}.name);
    fpath = fullfile(dst_folder, [fname,'.mat']);
    save(fpath,'feats');
end
caffe.reset();
end

function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
n = 1;
for i = indices
  for j = indices
    crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
    crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
    n = n + 1;
  end
end
center = floor(indices(2) / 2) + 1;
crops_data(:,:,:,5) = ...
  im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
end
