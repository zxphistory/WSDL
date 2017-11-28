% multifold detector training via an easy to hard mode
clear;clc

addpath utils
%% para settings
opts.network      = 'vgg-f';
opts.cache_name   = 'VOC2007';
opts.datainfo     = ['/home/zxphistory/data/' opts.cache_name  '/datainfo'];

load (opts.datainfo)
load (opts.region_pp) 
opts.datainfo     = datainfo;
opts.boxes    = boxes;

%% cache pool5 features
%--------------------------------------------------------------
% rcnn_cache_pool5_features(opts);
% rcnn_cache_pool5_neg_features(opts);
%% obtain postive features
% object-level positives, fc7 feats
opts.layer   = 5;
opts.ratio   = 0.01;
% opts.net_def_file = './model-defs/vgg_vd_pool5.prototxt';
% init model
rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file, opts);
rcnn_model = rcnn_load_model(rcnn_model, true);
% opts.pos = ['./pos_samples/pos_cub_obj_caffenet'];
opts.datainfo.path=opts.datainfo.path(opts.datainfo.trn);
% pos = rcnn_pos_features(opts, rcnn_model);

% part-level positives, pool5 feats
% opts.ratio   = 0.05;
% numcluster = [4 5];
% for num= numcluster
opts.cluster=2;
opts.pos = ['./pos_samples/pos_' opts.cache_name '_' num2str(opts.cluster) '_caffenet'];
pos = rcnn_pos_features(opts, rcnn_model);

% combine obj and part level positive features
% pos=[obj_pos; part_pos];

%% train detector
opts.crop_mode       = 'warp';
opts.crop_padding    =16;
opts.k_folds         = 0;
opts.layer           = 5;
opts.svm_C           = 10^-3;
opts.bias_mult       = 10;
opts.pos_loss_weight = 2 ;
opts.checkpoint      = 0 ;

% select a subset;
imdb_path           = ['imdb/data/imdb_' opts.cache_name '.mat'];
load (imdb_path)
imdb.class_ids      = 1;
imdb.classes        = 1;

% train detector
opts.cluster=2;
for i=1:length(pos)
  opts.part=i;
  opts.feat=pos{i};
  rcnn_train(imdb, opts);
end

%% visulization
% imdb.pos = pos{4};
% image_path='./vis/image/iter_samples/';
% round=1;
% image_ids=imdb.image_ids(opts.datainfo.trn);
% image_ids=image_ids(imdb.pos);
% for i=1:20
%    im=load_image(opts,image_ids{i});
%    sub_im=im(imdb.pos(i,3):imdb.pos(i,5),imdb.pos(i,2):imdb.pos(i,4),:);
%   imname=[image_path 'part_4_round' num2str(round) '_' num2str(i) '.png'];
%   imwrite(sub_im,imname,'png');
% end
% image_ids = imdb.image_ids(opts.datainfo.trn);
% dets=cell(3,1);
% part=[1 4 6];
% chs=[343 4 92];
% round=[11 11 3];
% for i=1:3
% dets{i}=load(['/home/zxphistory/code/rcnn/cachedir/cub_200/solver_state_part_' num2str(part(i)) '_round_' num2str(round(i)) '.mat']);
% end
% scores=zeros(length(dets{1}.rps),1);
% for i=1:length(scores)
%    scores(i)=max(dets{1}.rps{i})+max(dets{2}.rps{i})+max(dets{3}.rps{i});
% end
% 
% [~,im_id]=sort(scores,'descend');
% scores=scores(im_id);
% image_path='./vis/image/det_samples/';
% im_id=im_id';
% rps_path='/home/zxphistory/code/rcnn/data/domainNet/CUB-200-2011-rps_conv4_det/';
% for i=im_id([1:10:1000 5000:10:end])
%     im=load_image(opts,image_ids{i});
%     box=[];
%     box_rps=[];
%     rps_file=fullfile(rps_path,image_ids{i});
%     load (rps_file)
%    for num=1:3 
%     [~,idx]=max(dets{num}.rps{i});
%     box=[box; dets{num}.boxes{i}(idx,:)];
%     [~,idx]=max(d.feat(:,chs(num)));
%     box_rps=[box_rps; d.boxes(idx,:)];
%    end
%    showboxes(im,box);
%    imname=[image_path 'sample_' num2str(i) '.png'];
%    export_fig (imname, '-png')
%    showboxes(im,box_rps);
%    imname=[image_path 'sample_' num2str(i) '_rps.png'];
%    export_fig (imname, '-png')
% end


%% visulize flow chart 
% image_ids = imdb.image_ids(opts.datainfo.trn);
% dets=cell(3,1);
% part=[1 4 6];
% chs=[343 4 92];
% round=[11 11 3];
% sa_path='/home/zxphistory/data/CUB-200-2011/saliency/';
% for i=1:3
% dets{i}=load(['/home/zxphistory/code/rcnn/cachedir/cub_200/solver_state_part_' num2str(part(i)) '_round_' num2str(round(i)) '.mat']);
% end
% scores=zeros(length(dets{1}.rps),1);
% for i=1:length(scores)
%    scores(i)=max(dets{1}.rps{i})+max(dets{2}.rps{i})+max(dets{3}.rps{i});
% end
% 
% [~,im_id]=sort(scores,'descend');
% scores=scores(im_id);
% image_path='./vis/image/flow_chart/';
% im_id=im_id';
% for i=im_id(1:20)
%     im=load_image(opts,image_ids{i});
%     imname=[image_path 'im_' num2str(i) 'part_0.png'];
%     imwrite(im,imname,'png');
%     map=imread([sa_path,image_ids{i} '.jpg']);
%     if size(map,1)>size(map,2)
%       ratio=opts.minSize/size(map,2);
%       map=imresize(map,[ratio*size(map,1),opts.minSize],'bilinear','antialiasing',false);
%     else
%       ratio=opts.minSize/size(map,1);
%       map=imresize(map,[opts.minSize,ratio*size(map,2)],'bilinear','antialiasing',false);      
%     end
% %     subplot(2,2,1)
% %     imshow(im,[])
%    for num=1:3
%     box=[];
%     [~,idx]=sort(dets{num}.rps{i},'descend');
%     box=[box; [dets{num}.boxes{i}(idx(1:5),:) dets{num}.rps{i}(idx(1:5),:)]];
%     [gray, objHeatMap] = computeObjectnessHeatMap(im, box);
%     
%     map=mat2gray(map);
%     gray=mat2gray(gray.*map);
%     X = gray2ind(gray,256);
%     objHeatMap = ind2rgb(X,jet(256));
%     imname=[image_path 'im_' num2str(i) 'part_' num2str(num) '.png'];
%     imwrite(objHeatMap,imname,'png');
% %     subplot(2,2,num+1)
% %     imshow(objHeatMap,[])
%    end
% %    imname=[image_path 'sample_' num2str(i) '.png'];
% %    export_fig (imname, '-png')
% %    showboxes(im,box_rps);
% %    imname=[image_path 'sample_' num2str(i) '_rps.png'];
% %    export_fig (imname, '-png')
% end
%% direct based on conv4 features
% image_path='./vis/image/iter_samples/';
% round=11;
% for i=1:2:40
%     im=load_image(opts,image_ids{im_id(i)});
%     box=boxes{im_id(i)}(idx(i),:);
%     showboxes(im,box)
%     pause
% %     sub_im=im(box(2):box(4),box(1):box(3),:);
% %     imname=[image_path 'part_4_round' num2str(round) '_' num2str(i) '.png'];
% %     imwrite(sub_im,imname,'png');
% end