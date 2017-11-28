function warp_bboxes=warp_im(imsize,bboxes)
% warp image to include contxt and information...
crop_dim=224;
padd = 16;
warp_bboxes=[];
for i=1:size(bboxes,1)
 bbox=bboxes(i,:);
 height=bbox(4)-bbox(2)+1;
 width=bbox(3)-bbox(1)+1;
 if  height<width % resize height as crop_dim
    ratio=crop_dim/height;
    crop_height=ceil(padd*height./(crop_dim-padd*2));
    crop_width=ceil(padd*width./(width*ratio-padd*2));
 else                             % resize width as crop_dim
    ratio=crop_dim/width;
    crop_height=ceil(padd*height./(height*ratio-padd*2));
    crop_width=ceil(padd*width./(crop_dim-padd*2));
 end
  xmin=max(bbox(1)-crop_width,1);
  xmax=min(bbox(3)+crop_width,imsize(2));
  ymin=max(1,bbox(2)-crop_height);
  ymax=min(bbox(4)+crop_height,imsize(1));
  bbox=[xmin,ymin,xmax,ymax];
  warp_bboxes=[warp_bboxes;bbox];
end
end