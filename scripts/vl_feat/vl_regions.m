function d = vl_regions(net,im,boxes)
%compute CNN feats with single view

im_size = 224;
%mean substraction
im = single(im) ; % note: 255 range
image_mean=net.meta.normalization.averageImage;
im = im - imresize(image_mean, [size(im,1) size(im,2)],'bilinear','antialiasing',false);  

im_ = zeros(im_size, im_size, 3, size(boxes,1));
for t = 1:size(boxes,1)
  temp = im(boxes(t,2):boxes(t,4),boxes(t,1):boxes(t,3),:); 
  im_(:,:,:,t) = imresize(temp, [im_size, im_size],'bilinear','antialiasing',false);
end
 % convert to gpu mode
im_=gpuArray(single(im_));
res = vl_warpper(net, im_);
d.feat=squeeze(gather(res(end).x));
clear res;
d.feat=d.feat';
d.boxes = boxes;