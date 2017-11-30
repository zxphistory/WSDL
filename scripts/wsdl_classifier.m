function ap = wsdl_classifier(opts, imdb)

%load trained detectors
W =[];
B = [];
opts.feat_norm_mean = 0;
for c = 1:opts.numcategories
  conf.cache_dir = sprintf([opts.det_path '%d.mat'],c);
  load(conf.cache_dir);
  W  = [W cat(2,detectors.W{:})];
  B  = [B cat(2,detectors.B{:})];
end

feats = cell(length(imdb.images.name), 1);
for i=1:1:length(imdb.images.name)
  [catepath,imname]=fileparts(imdb.images.name{i});
  savefile=fullfile(opts.feat_cache, catepath, [imname '.mat']);
  d = load(savefile);
  f = Normalize(d.d.feat);
  s = max(f*W+B,[],1);
  feats{i} = Normalize(s);
  tic_toc_print('extracting features of %d finished.\n', i);
end

feats = cat(1,feats);
[ap,ap11] = classifier_voc(feats, imdb);
end