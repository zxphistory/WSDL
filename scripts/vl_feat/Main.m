%% baseline settings, extract features with original scale and perform classification
clear;clc
run 'matlab/vl_setupnn.m'
% parameter settings
opts.data_root     = 'D:/dataset/';
opts.dataset       = 'CUB-200-2011';
opts.path          = [opts.data_root opts.dataset '/images/'];
opts.datainfo      = [opts.data_root opts.dataset '/datainfo.mat'];
opts.exten         = [];
opts.numcategories = 200;
load (opts.datainfo);

% initial network
opts.scales=224;
opts.network='imagenet-vgg-verydeep-16';
Net=load (['models/' opts.network]);
% layer 31 33 35 --- pool5 fc6 fc7
opts.layer= 31;
Net.layers=Net.layers(1:opts.layer);
Net = vl_simplenn_move(Net, 'gpu');

% extract features
%mkdir_if_missing(['feats/' opts.dataset]);
featsfile=['feats/' opts.dataset '/feats_fixed_224'];
try 
  load(featsfile)
catch
  feats=cell(length(datainfo.path),1);
  for i=1:length(datainfo.path)
    impath=fullfile(opts.path, [datainfo.path{i} opts.exten]);
    im=imread(impath);
    if size(im,3)==1
      im=repmat(im,[1,1,3]);
    end
    d = vl_fixed(Net,im,opts.scales);
    feats{i}=d.feat;
    tic_toc_print('extracting features of %d finished.\n', i);
  end
  save(featsfile, 'feats', '-v7.3')
end
%% classification
fprintf('classification\n')
fprintf('-----------------------------------------\n')  
options = ['-c ' num2str(1) ', -s ' num2str(1), ', -q'];
trainingLabel = datainfo.labels(datainfo.trn, 2);
testingLabel = datainfo.labels(datainfo.tst, 2);
% 
org=false;
if org
  for i=1:length(feats)
    feats{i}=feats{i}(:);
  end
else
  for i=1:length(feats)
    f=feats{i}*feats{i}';
    feats{i}=sqrt(f(:));
  end  
end
feats=cat(2,feats{:});
feats=Normalize(feats');
featstrn=feats(datainfo.trn,:);
featstst=feats(datainfo.tst,:);
% featstrn=Normalize(featstrn);
% featstst=Normalize(featstst);
model = train(double(trainingLabel), sparse(double(featstrn)), options);
[testResult,~, decision_values] = predict(double(testingLabel), sparse(double(featstst)), model);
[cateAcc,imageAcc]=cpt_map(testResult, opts.numcategories, testingLabel);