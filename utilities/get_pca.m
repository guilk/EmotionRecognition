function [features] = get_pca(src_features, PC, means_norm, stds_norm)
% pca_file = '../../pca/generic_face_rigid.mat';
% load(pca_file);

dst_features = bsxfun(@times, bsxfun(@plus, src_features, -means_norm), 1./stds_norm);
features = dst_features * PC;
end