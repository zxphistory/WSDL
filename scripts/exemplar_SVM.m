function finished = exemplar_SVM(opts, DP)
% training exemplar SVM for each mined examples
%weak detectors learning

mkdir_if_missing(opts.exSVM_path);

rng(1);
% solver = 'liblinear';
opts.svm_C = 10^-3;
opts.max_neg = 10000;
opts.pos_loss_weight = opts.max_neg/2;
ll_opts = sprintf('-w1 %.5f -c %.5f -q',opts.pos_loss_weight, opts.svm_C);

for c = 1:length(DP)
  if ~exist([opts.exSVM_path num2str(c) '.mat'],'file') 
   patches = DP{c};
   neg = DP;
   neg{c} = [];
   neg = cat(1,neg{:});
   idx = randperm((size(neg,1)));
   neg = neg(idx(1:opts.max_neg),:);
   W = cell(size(patches,1),1);
   parfor i=1:size(patches,1)
     pos = patches(i,:);
     y = [1; -ones(size(neg,1),1)];
     X =  sparse(double([pos; neg]));
     llm = liblinear_train(y, X', ll_opts, 'col');  
     W{i} = llm.w;
   end
   fprintf('exemplar-SVM of category %d finished.\n', c);
   save([opts.exSVM_path num2str(c) '.mat'],'W','-v7.3');
  end
end
finished  = true;
end

