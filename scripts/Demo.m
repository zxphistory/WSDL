clear;clc
run 'matlab/vl_setupnn.m'
addpath scripts/vl_feat
addpath scripts/utils
addpath scripts/pascal

%--------------------------------------------------------------------------
%                                 demo classfication
opts.imdb_eval = './data/imdb_2007_eval.mat';
opts.numcategories = 20;
load(opts.imdb_eval);
% caffenet features
opts.feat_cache = './feats/VOC2007/30_reg_caffenet.mat';
load(opts.feat_cache);
feats=cat(1,feats{:});
ap_caffenet = classifier_voc(feats, imdb);
%vgg vd features
opts.feat_cache = './feats/VOC2007/30_reg_vd.mat';
load(opts.feat_cache);
feats=cat(1,feats{:});
ap_vd = classifier_voc(feats, imdb);
%--------------------------------------------------------------------------
