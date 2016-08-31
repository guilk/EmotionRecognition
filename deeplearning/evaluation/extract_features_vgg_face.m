function extract_features_vgg_face(net, im_list, dst_folder)

for im_index = 1:numel(im_list)
    feats = [];
    disp(im_list{im_index});
    I = imread(im_list{im_index});
    input_data = {prepare_image(I)};
    scores = net.forward(input_data);
    feats.fc7 = net.blobs('fc7').get_data();
    
    [~,fname] = fileparts(im_list{im_index});
    fpath = fullfile(dst_folder, [fname,'.mat']);
    save(fpath,'feats');
end

end

% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
mean_pix = [129.1863,104.7624,93.5940];
% IMAGE_DIM = 256;
CROPPED_DIM = 224;

% resize to fixed input size
im = single(im);

if size(im, 1) < size(im, 2)
    im = imresize(im, [CROPPED_DIM NaN],'bilinear');
else
    im = imresize(im, [NaN CROPPED_DIM],'bilinear');
end
if size(im,3) == 1
    im = cat(3,im,im,im);
end
% RGB -> BGR
im = im(:, :, [3 2 1]);
averageImg = mean_pix([3 2 1]);

images = cat(3,im(:,:,1)-averageImg(1),...
    im(:,:,2)-averageImg(2),...
    im(:,:,3)-averageImg(3));


% % oversample (4 corners, center, and their x-axis flips)
% images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
% 
% indices_y = [0 size(im,1)-CROPPED_DIM] + 1;
% indices_x = [0 size(im,2)-CROPPED_DIM] + 1;
% center_y = floor(indices_y(2) / 2)+1;
% center_x = floor(indices_x(2) / 2)+1;
% 
% curr = 1;
% for i = indices_y
%     for j = indices_x
%         images(:, :, :, curr) = ...
%             permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
%         images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
%         curr = curr + 1;
%     end
% end
% images(:,:,:,5) = ...
%     permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:), ...
%     [2 1 3]);
% images(:,:,:,10) = images(end:-1:1, :, :, curr);
% 
% % mean BGR pixel subtraction
% for c = 1:3
%     images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
% end

end