function imdb = setupVoc(datasetDir, varargin)
% SETUPVOC    Setup PASCAL VOC data
%   IMDB = SETUPVOC(DATASETDIR, 'EDITION', '2007') setups the
%   PASCAL VOC 2007 data. This is similar to SETUPGENERIC(), but adapted
%   to the peculiarities of this dataset. In particular, the
%   difficult image flag and the fact that multiple labels apply to
%   each image are supported.
%
%   Note that only the PASCAL VOC 2007 data comes with test images
%   and labels. For the other editions, setting up the test images
%   cannot be automatized due to restrictions in the distribution.
%
%   See also: SETUPGENERIC().

% Author: Andrea Vedaldi

% Copyright (C) 2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.edition = '2007' ;
opts = vl_argparse(opts, varargin) ;

imdb.images.id = [] ;
imdb.images.set = uint8([]) ;
imdb.images.name = {} ;
imdb.images.size = zeros(2,0) ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', ...
  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', ...
  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'} ;
imdb.imageDir = fullfile(datasetDir, ['VOC', opts.edition], 'JPEGImages') ;

% Get the list of images
map = containers.Map() ;
j = 0 ;
for si = 1:numel(imdb.meta.sets)
  setName = imdb.meta.sets{si} ;
  annoPath = fullfile(datasetDir, ['VOC' opts.edition], ...
    'ImageSets', 'Main', ...
    [setName '.txt']) ;
  %fprintf('%s: reading %s\n', mfilename, annoPath) ;
  [names,labels] = textread(annoPath, '%s %f') ;
  for i=1:length(names)
    j = j + 1 ;
    map(names{i}) = j ;
    imdb.images.id(j) = j ;
    imdb.images.set(j) = si ;
    imdb.images.name{j} = [names{i} '.jpg'] ;
    %info = imfinfo(fullfile(imdb.imageDir, imdb.images.name{j})) ;
    %imdb.images.size(:,j) = [info.Width ; info.Height] ;
    %fprintf('\radded %s', imdb.images.name{j}) ;
  end
end

% Get class labels
for ci = 1:length(imdb.meta.classes)
  imdb.classes.imageIds{ci} = [] ;
  imdb.classes.difficult{ci} = false(0) ;
end
for si = 1:numel(imdb.meta.sets)
  for ci = 1:length(imdb.meta.classes)
    setName = imdb.meta.sets{si} ;
    className = imdb.meta.classes{ci} ;
    annoPath = fullfile(datasetDir, ['VOC' opts.edition], ...
      'ImageSets', 'Main', ...
      [className '_' setName '.txt']) ;
   % fprintf('%s: reading %s\n', mfilename, annoPath) ;
    [names,labels] = textread(annoPath, '%s %f') ;
    for i = 1:numel(names)
      j = map(names{i}) ;
      if labels(i) >= 0
        imdb.classes.imageIds{ci}(end+1) = j ;
        imdb.classes.difficult{ci}(end+1) = (labels(i) == 0) ;
      end
    end
  end
end