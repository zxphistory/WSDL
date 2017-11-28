function d = vl_crosslayer(net, im, opts)
%compute CNN feats with multiple views

%mean substraction
im = single(im) ; % note: 255 range
d.height=size(im,1);d.width=size(im,2);
d.feat=cell(length(opts.scales)*2,1);
num=1;
image_mean=net.meta.normalization.averageImage;
image_mean=reshape(image_mean, [1 1 3]);
for s= opts.scales    
  if size(im,1)<size(im,2)
    ratio=s/size(im,1);
    im_ = imresize(im,[s ratio*size(im,2)],'bilinear','antialiasing',false);
  else
    ratio=s/size(im,2);
    im_ = imresize(im,[ratio*size(im,1),s],'bilinear','antialiasing',false);
  end
  im_ = im_ - repmat(image_mean, [size(im_,1) size(im_,2) 1]);
% convert to gpu mode
  if size(im_, 1)*size(im_, 2)<200000
    im_=gpuArray(im_);
    res = vl_warpper(net, im_);
    for layer = opts.crosslayer
      f=gather(res(layer).x);
      f=squeeze(mean(mean(f,1),2));
      f=f./norm(f,2);
      d.feat{num}=[d.feat{num};f];
    end
    clear res
    % flip
    res = vl_warpper(net, fliplr(im_));
    for layer = opts.crosslayer
      f=gather(res(layer).x);
      f=squeeze(mean(mean(f,1),2));
      f=f./norm(f,2);
      d.feat{num+1}=[d.feat{num+1};f];
    end
    clear res
  else
    net = vl_simplenn_move(net, 'cpu');
    res = vl_warpper(net, im_);
    for layer = opts.crosslayer
      f=res(layer).x;
      f=squeeze(mean(mean(f,1),2));
      f=f./norm(f,2);
      d.feat{num}=[d.feat{num};f];
    end
    clear res
    % flip
    res = vl_warpper(net, fliplr(im_));
    for layer = opts.crosslayer
      f=res(layer).x;
      f=squeeze(mean(mean(f,1),2));
      f=f./norm(f,2);
      d.feat{num+1}=[d.feat{num+1};f];
    end
    clear res
    net = vl_simplenn_move(net, 'gpu');
  end
  num=num+2;
end

d.feat=cat(2,d.feat{:});
d.feat=mean(d.feat,2);