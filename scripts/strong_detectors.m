function strong_detectors(opts, imdb, X_pos)
% easy to hard multifold MIL training for positive samples mining

opts.svm_C = 10^-3;
opts.bias_mult = 10;
opts.det_ids = 1:opts.num_clusters;
opts.class_ids  = 1:20;
opts.pos_loss_weight = 2;
opts.iter           = 3; 
opts.feat_norm_mean  = 0;
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

conf.file_path  = sprintf('./model-cache/mil/%s/',opts.dataset);
mkdir_if_missing(conf.file_path);

% iteratively train detectors
imdb.class_ids = opts.det_ids;
parfor c = 1:20
  detectors = process_category(opts, conf, imdb, X_pos, c);
  fprintf('training detectors of category %d finished.\n',c);
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function detectors = process_category(opts, conf, imdb, X_pos, c)
%--------------------------------------------------------------------------
trainval = imdb.images.set~=3;
conf.cache_dir = sprintf([conf.file_path '%d.mat'],c);
 if ~exist(conf.cache_dir, 'file')
  X_pos_c = X_pos{c};
  detectors.W = [];
  detectors.B = [];
  image_ids = imdb.images.name(imdb.images.label(c,trainval)~=1);
  caches = {};
  for j=1:length(X_pos_c)
   % Init training caches
    X_pos_c{j} = rcnn_scale_features(X_pos_c{j}, opts.feat_norm_mean);
    caches{j} = init_cache(X_pos_c{j});
  end
   % ----------------------------------------------------------------------
   % Train with hard negative mining
   first_time = true;
   max_hard_epochs = opts.iter;

  for hard_epoch = 1:max_hard_epochs
    for i = 1:length(image_ids)
      fprintf('%s: category %s hard neg epoch: %d/%d image: %d/%d\n', ...
            procid(), imdb.classes.name{c}, hard_epoch, max_hard_epochs, i, length(image_ids));

    % Get hard negatives for all classes at once (avoids loading feature cache
    % more than once)
     [X, keys] = sample_negative_features(first_time, caches, detectors,...
        image_ids, i, opts);

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
      is_last_time = (hard_epoch == max_hard_epochs && i == length(image_ids));
      hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
      if (first_time || hit_retrain_limit || is_last_time) && ...
          ~isempty(caches{j}.X_neg)
%         fprintf('>>> Updating %s detector <<<\n', imdb.classes.name{j});
%         fprintf('Cache holds %d pos examples %d neg examples\n', ...
%                 size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
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

%         for t = 1:length(caches{j}.tot_loss)
%           fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
%                   t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
%                   caches{j}.neg_loss(t), caches{j}.reg_loss(t));
%         end

        % store negative support vectors for visualizing later
%         SVs_neg = find(z_neg > -1 - eps);
%         rcnn_model.SVs.keys_neg{j} = caches{j}.keys_neg(SVs_neg, :);
%         rcnn_model.SVs.scores_neg{j} = z_neg(SVs_neg);

        % evict easy examples
        easy = find(z_neg < caches{j}.evict_thresh);
        caches{j}.X_neg(easy,:) = [];
        caches{j}.keys_neg(easy,:) = [];
%         fprintf('  Pruning easy negatives\n');
%         fprintf('  Cache holds %d pos examples %d neg examples\n', ...
%                 size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
%         fprintf('  %d pos support vectors\n', numel(find(z_pos <  1 + eps)));
%         fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
      end
    end
      first_time = false;
    end
   pos_image_ids = imdb.images.name(imdb.images.label(c,trainval)==1);
   [X_pos_c, keys_c, caches] = mining_pos(opts, pos_image_ids, detectors);  
  end
  save(conf.cache_dir, 'detectors', '-v7.3');
 end
 
% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, caches,...
                                                  detectors, image_ids, ind, opts)
% ------------------------------------------------------------------------
[~,imname] = fileparts(image_ids{ind});        
d = load([opts.feat_cache, imname '.mat']);
d=d.d;
class_ids = opts.det_ids;
if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);

if first_time
  for cls_id = class_ids 
    X_neg{cls_id} = d.feat;
    keys{cls_id} = [ind*ones(size(d.feat,1),1) [1:size(d.feat,1)]'];  
  end
else
  zs = bsxfun(@plus, d.feat*detectors.W, detectors.B);
  for cls_id = class_ids
    z = zs(:, cls_id);
    I = find((z > caches{cls_id}.hard_thresh));

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
   ll_opts = sprintf('-w1 %.5f -c %.5f -s %d -B %.5f, -q', ...
                      opts.pos_loss_weight, opts.svm_C, ...
                      liblinear_type, opts.bias_mult);
    %fprintf('liblinear opts: %s\n', ll_opts);
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


%-------------------------------------------------------------------------
function [X_pos, keys, caches] = mining_pos(opts, image_ids, detectors) 

X_pos = cell(numel(opts.det_ids),1);
keys = cell(numel(opts.det_ids),1);
class_ids = opts.det_ids;
for ind=1:length(image_ids)
 tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), ind, length(image_ids));
   
 [~,imname] = fileparts(image_ids{ind});            
 d = load([opts.feat_cache, imname '.mat']);
 d=d.d;

 d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
 
 zs = bsxfun(@plus, d.feat*detectors.W, detectors.B);
 for cls_id = class_ids
    z = zs(:, cls_id);
    [z,I] = max(z);
  % Unique hard negatives
  if z>0
   X_pos{cls_id} = [X_pos{cls_id}; d.feat(I,:)];
   keys{cls_id} = [keys{cls_id}; ind*ones(length(I),1) I];
  end
  end
end

caches = {};
for j=1:length(X_pos)
 % Init training caches
  X_pos{j} = rcnn_scale_features(X_pos{j}, opts.feat_norm_mean);
  caches{j} = init_cache(X_pos{j});
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
function cache = init_cache(X_pos)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = single([]);
cache.keys_neg = [];
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];
