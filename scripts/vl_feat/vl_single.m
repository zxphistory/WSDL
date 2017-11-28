function d = vl_single(net,im,minSize)
%compute CNN feats with single view

%mean substraction
im = single(im) ; % note: 255 range
d.height=size(im,1);d.width=size(im,2);
for s=minSize
    
    if size(im,1)<size(im,2)
       ratio=s/size(im,1);
       im_ = imresize(im,[s ratio*size(im,2)],'bilinear','antialiasing',false);
    else
       ratio=s/size(im,2);
       im_ = imresize(im,[ratio*size(im,1),s],'bilinear','antialiasing',false);
    end
    image_mean=net.meta.normalization.averageImage;
    im_ = im_ - repmat(image_mean, [size(im_,1) size(im_,2) 1]);
 % convert to gpu mode
 if size(im_, 1)*size(im_, 2)<200000
    im_=gpuArray(im_);
    res = vl_warpper(net, im_);
    d.feat=gather(res(end).x);
    clear res
 else
    net = vl_simplenn_move(net, 'cpu');
    res = vl_warpper(net, im_);
    d.feat=res(end).x;
    net = vl_simplenn_move(net, 'gpu');
    clear res
 end
    d.feat=mean(mean(d.feat,1),2);
end
% flip
im=fliplr(im);
for s=minSize
    
    if size(im,1)<size(im,2)
       ratio=s/size(im,1);
       im_ = imresize(im,[s ratio*size(im,2)],'bilinear','antialiasing',false);
    else
       ratio=s/size(im,2);
       im_ = imresize(im,[ratio*size(im,1),s],'bilinear','antialiasing',false);
    end
    image_mean=net.meta.normalization.averageImage;
    im_ = im_ - repmat(image_mean, [size(im_,1) size(im_,2) 1]);
 % convert to gpu mode
 if size(im_, 1)*size(im_, 2)<200000
    im_=gpuArray(im_);
    res = vl_warpper(net, im_);
    d.feat_flip=gather(res(end).x);
    clear res
 else
    net = vl_simplenn_move(net, 'cpu');
    res = vl_warpper(net, im_);
    d.feat_flip=res(end).x;
    net = vl_simplenn_move(net, 'gpu');
    clear res
 end
 
    d.feat_flip=mean(mean(d.feat_flip,1),2);
end

d.feat=mean([d.feat; d.feat_flip],1);