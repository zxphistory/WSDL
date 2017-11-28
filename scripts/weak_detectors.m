function X_pos = weak_detectors(opts, DP)
%initial pattern mining with detector based clustering

X_pos = cell(opts.numcategories, 1);

for c = 1:length(X_pos)
  dp = DP{c};
  % perform spectral clustering
  load([opts.exSVM_path num2str(c) '.mat']);
  W = cat(1,W{:});
  CMat = 1 - pdist2(W,W,'cosine');
  CKSym = BuildAdjacency(CMat);
  groups = SpectralClustering(CKSym,opts.num_clusters);
  pos = cell(max(groups),1);
  for i=1:max(groups)
    w = W(groups==i,:);
    s = sum(dp*w',2);
    [~, idx] = sort(s,'descend');
    num_patch = min(opts.patch_per_det, nnz(s>0));
    pos{i} = dp(idx(1:num_patch),:);
  end
  X_pos{c} = pos;
end
end