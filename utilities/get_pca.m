function [features] = get_pca(src_features, PC, means_norm, stds_norm)

dst_features = bsxfun(@times, bsxfun(@plus, src_features, -means_norm), 1./stds_norm);
features = dst_features * PC;

end