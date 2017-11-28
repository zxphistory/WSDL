function neg = rcnn_neg_features(opts)
% obtain negative samples only once

cache_name=[opts.cache_name '_neg' opts.network];
datainfo=opts.datainfo;
datainfo.path=datainfo.path(datainfo.trn);
neg=cell(length(opts.datainfo.path),1);

for i=1:1:length(neg)
   [categoryname,imname]= fileparts(datainfo.path{i});
   d = rcnn_load_cached_pool5_features(cache_name, [categoryname, '/' imname]);
   neg{i}=d.feat;
end 
end

