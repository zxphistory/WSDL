clear;clc
run 'matlab/vl_setupnn.m'
addpath scripts/vl_feat
addpath scripts/utils
addpath scripts/pascal

%--------------------------------------------------------------------------
%                                       parameter settings
opts.data_root     = '/home/zxphistory/data/'; 
opts.dataset       = 'VOC2007';
opts.exten         = '.jpg';
opts.numcategories = 20;
opts.dim           = 4096;
opts.dataDir       = [opts.data_root opts.dataset];
opts.proposalDir   = 'data/';
opts.imdbPath      = 'data/imdb_2007.mat';
opts.network       = 'imagenet-vgg-f';
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%                                     cache rcnn features
% finished = rcnn_cache_features(opts);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%                                   weak detector learning
DP = patch_mining(opts);

opts.exSVM_path = 'model-cache/exSVM_2007/';
finished = exemplar_SVM(opts,DP);

opts.num_clusters = 20;
opts.patch_per_det = 100;
X_pos = weak_detectors(opts, DP);
clear DP
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
                                  strong detector learning
imdb = load (opts.imdbPath);
opts.feat_cache = ['feats/' opts.dataset '/' opts.network '/'];
strong_detectors(opts, imdb, X_pos);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%                                 perform classfication
opts.det_path  = sprintf('./model-cache/mil/%s/',opts.dataset);
opts.imdb_eval = './data/imdb_2007_eval.mat';
load(opts.imdb_eval);
ap = wsdl_classifier(opts, imdb);
%--------------------------------------------------------------------------