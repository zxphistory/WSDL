function rcnn_e2h_train(imdb_o, roidb, opts)
% easy to hard multifold MIL training for positive samples mining

opts.svm_C = 10^-3;
opts.bias_mult = 10;
opts.class_ids = 1:20;
opts.pos_loss_weight = 2;
opts.kfold           = 5; 
opts.feat_norm_mean  = 0;
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

conf.file_path  = sprintf('./mil/feat_cache/%s/%s/', opts.model,opts.devkit);
mkdir_if_missing(conf.file_path);

% iteratively train detectors with easy to hard mode
for epoch=1
  % update imdb info, select a subset samples for training
  opts.iter = epoch;
  imdb = cnn_e2h_setup_data(imdb_o, opts);
  % image batches for current iteration
  prev_rng = seed_rand();
  trainval = find(imdb.images.set==1 | imdb.images.set==2);
  image_ids = imdb.images.name(trainval);
  image_ids = image_ids(randperm(length(image_ids)));
  rng(prev_rng);
  detectors.W = [];
  detectors.B = [];
 for k=1:opts.kfold
  % multifold cross validation
  conf.cache_dir = sprintf([conf.file_path 'detectors_epoch-%d_fold-%d.mat'],opts.iter,k);
  try
      load(conf.cache_dir)
  catch
  test_ids = zeros(length(image_ids),1);
  test_ids(k:opts.kfold:end) = 1; % one fold for test
  trainval_ids = ~test_ids;
  % load positive features, we load it only once
  save_file = sprintf('./mil/feat_cache/%s/%s/gt_pos_cache_epoch-%d_fold-%d.mat', ...
      opts.model,opts.devkit,epoch,k);
  try
    load(save_file);
    fprintf('Loaded saved positives from mined ground truth boxes\n');
  catch
    [X_pos, keys_pos] = get_positive_features(imdb, roidb, image_ids(trainval_ids), opts);
    save(save_file, 'X_pos', 'keys_pos', '-v7.3');
  end
% Init training caches
caches = {};
for i = imdb.class_ids
  fprintf('%14s has %6d positive instances\n', ...
      imdb.classes.name{i}, size(X_pos{i},1));
  X_pos{i} = rcnn_scale_features(X_pos{i}, opts.feat_norm_mean);
  caches{i} = init_cache(X_pos{i}, keys_pos{i});
end

% ------------------------------------------------------------------------
% Train with hard negative mining
first_time = true;
% one pass over the data is enough
max_hard_epochs = 1;

for hard_epoch = 1:max_hard_epochs
  for i = 1:length(image_ids(trainval_ids))
    fprintf('%s: hard neg epoch: %d/%d image: %d/%d\n', ...
            procid(), hard_epoch, max_hard_epochs, i, length(image_ids(trainval_ids)));

    % Get hard negatives for all classes at once (avoids loading feature cache
    % more than once)
    [X, keys] = sample_negative_features(first_time, roidb, caches, detectors,...
        image_ids(trainval_ids), i, opts);

    % Add sampled negatives to each classes training cache, removing
    % duplicates
    for j = imdb.class_ids
      if ~isempty(keys{j})
        if ~isempty(caches{j}.keys_neg)
          [~, ~, dups] = intersect(caches{j}.keys_neg, keys{j}, 'rows');
          assert(isempty(dups));
        end
        caches{j}.X_neg = cat(1, caches{j}.X_neg, X{j});
        caches{j}.keys_neg = cat(1, caches{j}.keys_neg, keys{j});
        caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
      end

      % Update model if
      %  - first time seeing negatives
      %  - more than retrain_limit negatives have been added
      %  - its the final image of the final epoch
      is_last_time = (hard_epoch == max_hard_epochs && i == length(image_ids(trainval_ids)));
      hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
      if (first_time || hit_retrain_limit || is_last_time) && ...
          ~isempty(caches{j}.X_neg)
        fprintf('>>> Updating %s detector <<<\n', imdb.classes.name{j});
        fprintf('Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
        [new_w, new_b] = update_model(caches{j}, opts);
        detectors.W(:, j) = new_w;
        detectors.B(j) = new_b;
        caches{j}.num_added = 0;

        z_pos = caches{j}.X_pos * new_w + new_b;
        z_neg = caches{j}.X_neg * new_w + new_b;

        caches{j}.pos_loss(end+1) = opts.svm_C * opts.pos_loss_weight * ...
                                    sum(max(0, 1 - z_pos));
        caches{j}.neg_loss(end+1) = opts.svm_C * sum(max(0, 1 + z_neg));
        caches{j}.reg_loss(end+1) = 0.5 * new_w' * new_w + ...
                                    0.5 * (new_b / opts.bias_mult)^2;
        caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                    caches{j}.neg_loss(end) + ...
                                    caches{j}.reg_loss(end);

        for t = 1:length(caches{j}.tot_loss)
          fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                  t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                  caches{j}.neg_loss(t), caches{j}.reg_loss(t));
        end

        % store negative support vectors for visualizing later
%         SVs_neg = find(z_neg > -1 - eps);
%         rcnn_model.SVs.keys_neg{j} = caches{j}.keys_neg(SVs_neg, :);
%         rcnn_model.SVs.scores_neg{j} = z_neg(SVs_neg);

        % evict easy examples
        easy = find(z_neg < caches{j}.evict_thresh);
        caches{j}.X_neg(easy,:) = [];
        caches{j}.keys_neg(easy,:) = [];
        fprintf('  Pruning easy negatives\n');
        fprintf('  Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
        fprintf('  %d pos support vectors\n', numel(find(z_pos <  1 + eps)));
        fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
      end
    end
    first_time = false;
  end
end
% save current detector model
  save(conf.cache_dir, 'detectors', '-v7.3');
  end
 end
   % mine new examples based on cross validation detectors
  for k=1:opts.kfold
   % multifold cross validation
    conf.cache_dir = sprintf([conf.file_path 'detectors_epoch-%d_fold-%d.mat'],opts.iter,k);
  try
      load(conf.cache_dir)
  catch
    warning('no trained detectors available');    
  end
 
  test_ids = zeros(length(image_ids),1);
  test_ids(k:opts.kfold:end) = 1; % one fold for test
  mining_pos(imdb_o, image_ids(test_ids), detectors);
    end
end
% ------------------------------------------------------------------------



% ------------------------------------------------------------------------


%--------------------------------------------------------------------------
% ------------------------------------------------------------------------
function [X_pos, keys] = get_positive_features(imdb, roidb, image_ids, opts)
% ------------------------------------------------------------------------
X_pos = cell(max(imdb.class_ids), 1);
keys = cell(max(imdb.class_ids), 1);

for i = 1:length(image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(image_ids));
  [~,imname] = fileparts(image_ids{i});          
  idx = strcmp(roidb.imnames,imname);
  rois = roidb.rois{idx};
  d = load([opts.feat_cache, imname '.mat']);
  d=d.d;
  
  for j = imdb.class_ids
    if isempty(X_pos{j})
      X_pos{j} = single([]);
      keys{j} = [];
    end
    sel = find(rois.overlap(:,j) == 1);
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, d.feat(sel,:));
      keys{j} = cat(1, keys{j}, [i*ones(length(sel),1) sel]);
    end
  end
end
%--------------------------------------------------------------------------


% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, roidb, caches,...
                                                  detectors, image_ids, ind, opts)
% ------------------------------------------------------------------------
[~,imname] = fileparts(image_ids{ind});        
idx = strcmp(roidb.imnames,imname);
rois = roidb.rois{idx};
d = load([opts.feat_cache, imname '.mat']);
d=d.d;
class_ids = opts.class_ids;
if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
neg_ovr_thresh = 0.3;

if first_time
  for cls_id = class_ids
    I = find(rois.overlap(:, cls_id) < neg_ovr_thresh);
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
else
  zs = bsxfun(@plus, d.feat*detectors.W, detectors.B);
  for cls_id = class_ids
    z = zs(:, cls_id);
    I = find((z > caches{cls_id}.hard_thresh) & ...
             (rois.overlap(:, cls_id) < neg_ovr_thresh));

    % Avoid adding duplicate features
    keys_ = [ind*ones(length(I),1) I];
    if ~isempty(caches{cls_id}.keys_neg) && ~isempty(keys_)
      [~, ~, dups] = intersect(caches{cls_id}.keys_neg, keys_, 'rows');
      keep = setdiff(1:size(keys_,1), dups);
      I = I(keep);
    end

    % Unique hard negatives
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
end

% ------------------------------------------------------------------------
function [w, b] = update_model(cache, opts, pos_inds, neg_inds)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 3;  % l2 regularized l1 hinge loss
%liblinear_type = 5; % l1 regularized l2 hinge loss

if ~exist('pos_inds', 'var') || isempty(pos_inds)
  num_pos = size(cache.X_pos, 1);
  pos_inds = 1:num_pos;
else
  num_pos = length(pos_inds);
  fprintf('[subset mode] using %d out of %d total positives\n', ...
      num_pos, size(cache.X_pos,1));
end
if ~exist('neg_inds', 'var') || isempty(neg_inds)
  num_neg = size(cache.X_neg, 1);
  neg_inds = 1:num_neg;
else
  num_neg = length(neg_inds);
  fprintf('[subset mode] using %d out of %d total negatives\n', ...
      num_neg, size(cache.X_neg,1));
end

switch solver
  case 'liblinear'
    ll_opts = sprintf('-w1 %.5f -c %.5f -s %d -B %.5f', ...
                      opts.pos_loss_weight, opts.svm_C, ...
                      liblinear_type, opts.bias_mult);
    fprintf('liblinear opts: %s\n', ll_opts);
    X = sparse(size(cache.X_pos,2), num_pos+num_neg);
    X(:,1:num_pos) = cache.X_pos(pos_inds,:)';
    X(:,num_pos+1:end) = cache.X_neg(neg_inds,:)';
    y = cat(1, ones(num_pos,1), -ones(num_neg,1));
    llm = liblinear_train(y, X, ll_opts, 'col');
    w = single(llm.w(1:end-1)');
    b = single(llm.w(end)*opts.bias_mult);

  otherwise
    error('unknown solver: %s', solver);
end

%--------------------------------------------------------------------------

pos = mining_pos(imdb_o, image_ids(test_ids), detectors);
%--------------------------------------------------------------------------
% ------------------------------------------------------------------------
function [W, B, folds] = update_model_k_fold(rcnn_model, caches, imdb)
% ------------------------------------------------------------------------
opts = rcnn_model.training_opts;
num_images = length(imdb.image_ids);
folds = create_folds(num_images, opts.k_folds);
W = cell(opts.k_folds, 1);
B = cell(opts.k_folds, 1);

fprintf('Training k-fold models\n');
for i = imdb.class_ids
  fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
  fprintf('Training folds for class %s\n', imdb.classes(i));
  fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');
  for f = 1:length(folds)
    fprintf('Holding out fold %d\n', f);
    [pos_inds, neg_inds] = get_cache_inds_from_fold(caches{i}, folds{f});
    [new_w, new_b] = update_model(caches{i}, opts, ...
        pos_inds, neg_inds);
    W{f}(:,i) = new_w;
    B{f}(i) = new_b;
  end
end


% ------------------------------------------------------------------------
function [pos_inds, neg_inds] = get_cache_inds_from_fold(cache, fold)
% ------------------------------------------------------------------------
pos_inds = find(ismember(cache.keys_pos(:,1), fold) == false);
neg_inds = find(ismember(cache.keys_neg(:,1), fold) == false);

%-------------------------------------------------------------------------
function [rps, feats]=cpt_rps(imdb, rcnn_model, opts)
%-------------------------------------------------------------------------
if ~opts.trn   % last time trn as pos, tst as validation
  image_ids = imdb.image_ids(opts.datainfo.tst);  
else          %  last time tst as pos, trn as validation
  image_ids = imdb.image_ids(opts.datainfo.trn);  
end

rps=cell(length(image_ids),1);
feats=cell(length(image_ids),1);
feat_cache=[imdb.name '_' opts.network];
feat_mean=0;
W=rcnn_model.detectors.W;
B=rcnn_model.detectors.B;
layer=opts.layer;

for ind=1:length(image_ids)
 tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), ind, length(image_ids));
   
 d = rcnn_load_cached_pool5_features(feat_cache, image_ids{ind});
 d.feat = rcnn_pool5_to_fcX(d.feat, layer, rcnn_model);

 d.feat = rcnn_scale_features(d.feat, feat_mean);
 
 rps{ind} = bsxfun(@plus, d.feat*W, B);
 [~,idx]=max(rps{ind});
 feats{ind}=d.feat(idx,:);
end


%-------------------------------------------------------------------------
function X_pos=update_pos(rps, feats, imdb, opts)
%-------------------------------------------------------------------------
% X_pos are weighted version of features

if ~opts.trn   % last time trn as pos, tst as validation
  image_ids = imdb.image_ids(opts.datainfo.tst);
else          %  last time tst as pos, trn as validation
  image_ids = imdb.image_ids(opts.datainfo.trn);
end

pos_det_thre   = 0.9;
feat_norm_mean = 0;
scores=zeros(length(rps),1);
for i=1:length(scores)
   scores(i)=max(rps{i});
end

% record valid idx
[~,ind]=sort(scores,'descend');
top_rank=round(opts.ratio*length(scores));
top_rank=max(top_rank,nnz(scores>pos_det_thre));
ind=ind(1:top_rank);
scores=scores(ind);
feats=feats(ind);
image_ids=image_ids(ind);

X_pos=cell(length(ind),1);

for i=1:length(ind)
   tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(ind));
   X_pos{i} = rcnn_scale_features(feats{i}, feat_norm_mean, scores(i)); 
%    im=load_image(opts, [image_ids{i},'.jpg']);
%    showboxes(im, d.boxes(idx(1),:));
%    fprintf('score %d.\n',scores(i))
%    pause
end
 
% ------------------------------------------------------------------------
function cache = init_cache(X_pos,keys_pos)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = single([]);
cache.keys_neg = [];
cache.keys_pos = keys_pos;
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];
