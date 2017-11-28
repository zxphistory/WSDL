function d = vl_mscale(net,im,minSize)
%compute CNN feats with multiple views

%mean substraction
im = single(im) ; % note: 255 range
d.height=size(im,1);d.width=size(im,2);
d.feat=cell(length(minSize)*2,1);
num=1;
image_mean=net.meta.normalization.averageImage;
if size(image_mean, 1)==1
  image_mean=reshape(image_mean, [1 1 3]);
end
for s=minSize    
  if size(im,1)<size(im,2)
    ratio=s/size(im,1);
    im_ = imresize(im,[s ratio*size(im,2)],'bilinear','antialiasing',false);
  else
    ratio=s/size(im,2);
    im_ = imresize(im,[ratio*size(im,1),s],'bilinear','antialiasing',false);
  end
  
  if size(image_mean,1)==1
    im_ = im_ - repmat(image_mean, [size(im_,1) size(im_,2) 1]);
  else
    im_ = im_ - imresize(image_mean, [size(im_,1) size(im_,2)]);  
  end
% convert to gpu mode
  if size(im_, 1)*size(im_, 2)<180000
    im_=gpuArray(im_);
    res = vl_warpper(net, im_);
    d.feat{num}=gather(res(end).x);
    clear res
    % flip
    res = vl_warpper(net, fliplr(im_));
    d.feat{num+1}=gather(res(end).x);
  else
    net = vl_simplenn_move(net, 'cpu');
    res = vl_warpper(net, im_);
    d.feat{num}=res(end).x;
    clear res
    % flip
    res = vl_warpper(net, fliplr(im_));
    d.feat{num+1}=res(end).x;
    clear res
    net = vl_simplenn_move(net, 'gpu');
  end
  num=num+2;
end

for i=1:length(d.feat)
    f=d.feat{i};
    f=permute(f, [3 1 2]);
    f=reshape(f, [size(f,1) size(f,2)*size(f,3)]);
    d.feat{i}=f;
end
d.feat=cat(2,d.feat{:});
d.feat=mean(d.feat,2);