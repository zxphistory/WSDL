function feats = rcnn_pos_features(opts, rcnn_model)
% obtain object and part level positive features
% pos: store the positve coordinates

ratio=opts.ratio;
cache_name=[opts.cache_name '_' opts.network];
datainfo=opts.datainfo;
% datainfo.path=datainfo.path(datainfo.trn);
load (opts.pos);

feats=[];
num= round(length(datainfo.path)*ratio);
im_cache='pos_samples/cub_obj/';
mkdir_if_missing(im_cache);
for k=1:length(pos)
 feat=cell(num,1);
 p=pos{k};
 for i=1:num
   [categoryname,imname]= fileparts(datainfo.path{p(i,1)});
%    im=imread(fullfile(opts.impath, categoryname, [imname '.jpg']));
%    if size(im,1)<size(im,2)
%     ratio=opts.minSize/size(im,1);
%     im = imresize(im,[opts.minSize ratio*size(im,2)],'bilinear','antialiasing',false);
%    else
%     ratio=opts.minSize/size(im,2);
%     im = imresize(im,[ratio*size(im,1),opts.minSize],'bilinear','antialiasing',false);
%    end
%    sub_im=im(p(i,3):p(i,5),p(i,2):p(i,4),:);
%    savefile=[im_cache imname '.jpg'];
%    imwrite(sub_im, savefile);
   d = rcnn_load_cached_pool5_features(cache_name, [categoryname, '/' imname]);
   diff=sum(d.boxes.^2,2)-sum(p(i,2:5).^2);
   idx=find(diff==0);
   feat{i} = rcnn_pool5_to_fcX(d.feat(idx,:), opts.layer, rcnn_model);
   tic_toc_print('proecssing %d finished.\n', i)
%   showboxes(im, boxes);
%   fprintf('score %.3f.\n',score(i))
%   pause
 end
feat=cat(1,feat{:});
feats=[feats;{feat}];
end
end

