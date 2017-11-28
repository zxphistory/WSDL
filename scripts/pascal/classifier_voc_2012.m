function [ap,ap11] = classifier_voc_2012(descrs, imdb, test_online)


descrs=Normalize(descrs);
descrs=descrs';
if isfield(imdb.images, 'class')
  classRange = unique(imdb.images.class) ;
else
  classRange = 1:numel(imdb.classes.imageIds) ;
end
numClasses = numel(classRange) ;

% trainval = find(imdb.images.set <= 2) ;
% test = find(imdb.images.set == 3) ;
% scores = cell(1, numel(classRange)) ;
% ap = zeros(1, numel(classRange)) ;
% ap11 = zeros(1, numel(classRange)) ;
% w = cell(1, numel(classRange)) ;
% for c = 1:numel(classRange)
%   if isfield(imdb.images, 'class')
%     y = 2 * (imdb.images.class == classRange(c)) - 1 ;
%   else
%     y = - ones(1, numel(imdb.images.id)) ;
%     [~,loc] = ismember(imdb.classes.imageIds{classRange(c)}, imdb.images.id) ;
%     y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
%   end
%   if all(y <= 0), continue ; end
% 
%   y(y==0)=1;
%   pos_loss=nnz(y(trainval)==-1)/nnz(y(trainval)==1);
%   options = ['-w1 ' num2str(pos_loss) ', -c ' num2str(1) ', -s ' num2str(1), ', -q'];
%   options = ['-c ' num2str(1) ', -s ' num2str(1), ', -q'];
%   model = train(y(trainval)', sparse(double(descrs(:,trainval)')), options) ;
%   w{c}=model.w;
%   scores{c} = w{c} * descrs ;
% 
%   [~,~,info] = vl_pr(y(test), scores{c}(test)) ;
%   ap(c) = info.ap ;
%   ap11(c) = info.ap_interp_11 ;
%   fprintf('class %s AP %.2f; AP 11 %.2f\n', imdb.meta.classes{classRange(c)}, ...
%           ap(c) * 100, ap11(c)*100) ;
% end


% train and test
opts.C=1;
trainval = find(imdb.images.set <= 1) ;
test = find(imdb.images.set == 2) ;
lambda = 1 / (opts.C*numel(trainval)) ;
par = {'Solver', 'sdca', 'Verbose', ...
       'BiasMultiplier', 1, ...
       'Epsilon', 0.001, ...
       'MaxNumIterations', 100 * numel(trainval)} ;

scores = cell(1, numel(classRange)) ;
ap = zeros(1, numel(classRange)) ;
ap11 = zeros(1, numel(classRange)) ;
w = cell(1, numel(classRange)) ;
b = cell(1, numel(classRange)) ;
for c = 1:numel(classRange)
  if isfield(imdb.images, 'class')
    y = 2 * (imdb.images.class == classRange(c)) - 1 ;
  else
    y = - ones(1, numel(imdb.images.id)) ;
    [~,loc] = ismember(imdb.classes.imageIds{classRange(c)}, imdb.images.id) ;
    y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
  end
  if all(y <= 0), continue ; end

  [w{c},b{c}] = vl_svmtrain(descrs(:,trainval), y(trainval), lambda, par{:}) ;
  scores{c} = w{c}' * descrs + b{c} ;

  if test_online
    im_ids = imdb.images.name(test);
    creat_result_file(w, b, im_ids);
  else
   [~,~,info] = vl_pr(y(test), scores{c}(test)) ;
   ap(c) = info.ap ;
   ap11(c) = info.ap_interp_11 ;
   fprintf('class %s AP %.2f; AP 11 %.2f\n', imdb.meta.classes{classRange(c)}, ...
          ap(c) * 100, ap11(c)*100) ;
  end
end

end


function creat_result_file(w, b, im_ids)
% create results file
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

fid=fopen(sprintf(VOCopts.clsrespath,'comp2',cls),'w');

% classify each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification
    c=classify(VOCopts,classifier,fd);
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file
fclose(fid);
end