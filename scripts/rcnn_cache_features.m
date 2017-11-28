function finished = rcnn_cache_features(opts)
% extract cnn features

try 
    imdb = load (opts.imdbPath);
catch
    imdb = cnn_voc07_eb_setup_data('dataDir',opts.dataDir, ...
            'proposalDir',opts.proposalDir,'loadTest',1);
    save(opts.imdbPath,'-struct', 'imdb', '-v7.3');
end

% initial network
Net=load (['models/' opts.network '.mat']);
% layer 17 --- fc6
opts.layer= 17;
Net.layers=Net.layers(1:opts.layer);
Net = vl_simplenn_move(Net, 'gpu');

% extract features
mkdir_if_missing(['feats/' opts.dataset]);
feat_cache = ['feats/' opts.dataset '/' opts.network];
mkdir_if_missing(feat_cache);
for i=1:1:length(imdb.images.name)
  [catepath,imname]=fileparts(imdb.images.name{i});
  mkdir_if_missing(fullfile(feat_cache, catepath));
  savefile=fullfile(feat_cache, catepath, [imname '.mat']);
  if ~exist(savefile,'file') 
   imfile=fullfile(imdb.imageDir, catepath, [imname '.jpg']);
   im=imread(imfile);
   if size(im,3)==1
     im=repmat(im,[1,1,3]);
   end
   boxes = imdb.images.boxes{i};
   % [x1 y1 x2 y2];
   boxes = boxes(:,[2 1 4 3]);
   boxes_ = warp_im(imdb.images.size(i,:),double(boxes));
   d = vl_regions(Net,im,boxes_);
   save(savefile, 'd', '-v7.3')
   tic_toc_print('extracting features of %d finished.\n', i);
  end
end
finished = true;
end

