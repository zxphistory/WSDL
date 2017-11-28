function feats =Normalize(feats, norm_mean)
%separate normalization
if nargin<2
    norm_mean=1;
end
power=2;
for i=1:size(feats,1)
     s = (sum(abs(feats(i,:).^ power)) .^ (1 / power));
     if s
       feats(i,:)=norm_mean*feats(i,:)/s;
     end
end
end

