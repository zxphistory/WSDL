function d = vl_fixed(net,im,minSize)
%compute CNN feats with multiple views

%mean substraction
im = single(im) ; % note: 255 range
d.height=size(im,1);d.width=size(im,2);
d.feat=cell(length(minSize)*2,1);
num=1;
image_mean=net.meta.normalization.averageImage;
image_mean=reshape(image_mean, [1 1 3]);
    
im_ = imresize(im,[minSize, minSize],'bilinear','antialiasing',false);
im_ = im_ - repmat(image_mean, [size(im_,1) size(im_,2) 1]);
% convert to gpu mode
  if size(im_, 1)*size(im_, 2)<200000
    im_=gpuArray(im_);
    res = vl_warpper(net, im_);
    d.feat{num}=gather(res(end).x);
    clear res
    % flip
%     res = vl_warpper(net, fliplr(im_));
%     d.feat{num+1}=gather(res(end).x);
  else
    net = vl_simplenn_move(net, 'cpu');
    res = vl_warpper(net, im_);
    d.feat{num}=res(end).x;
    clear res
    % flip
%     res = vl_warpper(net, fliplr(im_));
%     d.feat{num+1}=res(end).x;
%     clear res
%     net = vl_simplenn_move(net, 'gpu');
  end

for i=1:length(d.feat)
    f=d.feat{i};
    f=permute(f, [3 1 2]);
    f=reshape(f, [size(f,1) size(f,2)*size(f,3)]);
    d.feat{i}=f;
end
d.feat=d.feat{1};
% d.feat=cat(2,d.feat{:});
% d.feat=mean(d.feat,2);