function DP = patch_mining(opts)
% discriminative patch mining based on the svm model

opts.DP_path = 'model-cache/dp_2007.mat';
if exist(opts.DP_path,'file')
  load(opts.DP_path);
  return;
end

opts.svm_model = './model-cache/svm_model.mat';
opts.imdbPath    = 'data/imdb_2007_eval.mat';
opts.feat_cache = ['feats/' opts.dataset '/' opts.network];

try 
    load (opts.imdbPath)
catch
    imdb = setupVoc([opts.dataDir '/VOCdevkit'], 'edition', '2007') ;
    save(opts.imdbPath, 'imdb', '-v7.3');
end

try 
    load (opts.svm_model)
catch
  fprintf('loading features\n')
  fprintf('-----------------------------------------\n') 
  feats = cell(length(imdb.images.name), 1);
  for i=1:1:length(imdb.images.name)
    [catepath,imname]=fileparts(imdb.images.name{i});
    savefile=fullfile(opts.feat_cache, catepath, [imname '.mat']); 
    load (savefile);
    f = sum(d.feat,1);
    feats{i} = Normalize(f);
  end

  fprintf('classification\n')
  fprintf('-----------------------------------------\n') 
  feats = cat(1,feats{:});
  [ap,ap11, w, b] = classifier_voc(feats, imdb);
  fprintf('mAP: %.2f %%; mAP 11: %.2f\n', mean(ap) * 100, mean(ap11) * 100);
  save(opts.svm_model,'w','b','-v7.3');
end

% discriminaitve patch mining
opts.max_percategory = 10000;
opts.max_perim       = 100;
opts.min_perim       = 20;
opts.thre            = 2;
DP = cell(opts.numcategories, 1);
trainval = imdb.images.set <= 2 ;
for c = 1:length(DP)
  [~,loc] = ismember(imdb.classes.imageIds{c}, imdb.images.id(trainval)) ;
  idx = imdb.classes.difficult{c}==1;
  loc(idx) = 0;
  loc(loc==0) = [];
  dp = cell(numel(loc),1);
  for i =1:numel(loc)
    [catepath,imname]=fileparts(imdb.images.name{loc(i)});
    savefile=fullfile(opts.feat_cache, catepath, [imname '.mat']); 
    load (savefile);
    f = Normalize(d.feat);
    s = f*w{c}+b{c};
    s = log(1+exp(s));
    im=imread(fullfile(imdb.imageDir,imdb.images.name{loc(i)}));
    gray = computeObjectnessHeatMap(im,[d.boxes s]);
    boxes = d.boxes;
    sa = zeros(size(boxes,1),1);
    for t = 1:size(boxes,1)
      map = gray(boxes(t,2):boxes(t,4),boxes(t,1):boxes(t,3));
      sa(t) = sum(map(:))./numel(map(:));
    end
    scores = sa+s;
    [~,idx] = sort(scores,'descend');
    num_patch = max(opts.min_perim, min(sum(scores>opts.thre),opts.max_perim)); 
    dp{i} = f(idx(1:num_patch),:);
  end
  dp = cat(1,dp{:});
  if size(dp,1)>opts.max_percategory
    idx = randperm(size(dp,1));
    dp = dp(idx(1:opts.max_percategory),:);
  end
  DP{c} = dp;
  fprintf('mininig patches of category %d finished.\n', c);
end
  save(opts.DP_path,'DP','-v7.3');
end

