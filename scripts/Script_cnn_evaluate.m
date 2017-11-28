%evaluate the formance of cnn based resuts
clear;clc
addpath matlab
addpath scripts/pascal
imdb = load('/home/zxphistory/code/matconvnet/feats/VOC2007/imdb_test_scores_01.mat');

classRange = 1:numel(imdb.classes.name);
scores = cat(2,imdb.images.test_scores{:});
scores = scores';
test = find(imdb.images.set == 3) ;
for c = 1:numel(classRange)
  y = imdb.images.label(:,c); 
  [~,~,info] = vl_pr(y(test), scores(:, c)) ;
  ap(c) = info.ap ;
  ap11(c) = info.ap_interp_11 ;
  fprintf('class %s AP %.2f; AP 11 %.2f\n', imdb.classes.name{classRange(c)}, ...
          ap(c) * 100, ap11(c)*100) ;
end