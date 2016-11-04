%addpath('./record_codes/TNT_layers_aPy');
addpath('./TNT_layers');
dataset_list = {'AWA','aPy','CUB'}; 
feature_list = {'vgg','google_net'};
method = {'TNT','SynC','Simple'};
dataset = dataset_list{1};   
feature = feature_list{2};
source = []; target_train = []; target_test = [];
addpath('/home/chuna/Zeroshot/ES/Embarrassingly-simple-ZSL-master');
acc_all = [];

for k = 1:3
    for i = [1,3]
        dataset = dataset_list{i}; 
        data.dataset = dataset;    
        if(strcmp(dataset,'AWA'))
            load '/home/chuna/Zeroshot/SSE/cnn-features/AwA/predicateMatrixContinuous.mat' predicateMatrixContinuous     
            if(strcmp(feature,'vgg'))
                load '/home/chuna/Zeroshot/SSE/cnn-features/AwA/feat-imagenet-vgg-verydeep-19' train_feat train_labels test_feat test_labels;
                X_all = normc([train_feat,test_feat]);
                Y_all = [train_labels',test_labels'];
                non_zero_category = unique(train_labels');
                zero_category = unique(test_labels');
            elseif(strcmp(feature,'google_net'))
                load '/home/chuna/Zeroshot/SynC/data/AwA_googlenet' X;
                load '/home/chuna/Zeroshot/SynC/data/AWA_inform_release' tr_loc te_loc y_tr y_te;
                X_all = normc([X(tr_loc,:);X(te_loc,:)]');
                Y_all = [y_tr,y_te];
                non_zero_category = unique(y_tr);
                zero_category = unique(y_te);
            end

            attr_table = normc(double(predicateMatrixContinuous'));
 
        elseif(strcmp(dataset,'aPy'))
            load '/home/chuna/Zeroshot/SSE/cnn-features/aPY/cnn_feat_imagenet-vgg-verydeep-19.mat' cnn_feat
            load '/home/chuna/Zeroshot/SSE/cnn-features/aPY/class_attributes.mat' labels class_attributes
            
            X_all = normc(cnn_feat);
            Y_all = labels';
            non_zero_category = 1:20;
            zero_category = 21:32;
            attr_table = normc(double(class_attributes)');
                 
        elseif(strcmp(dataset,'CUB'))
            if(strcmp(feature,'vgg'))
                load '/home/chuna/Zeroshot/SSE/cnn-features/CUB_200_2011/cnn_feat-imagenet-vgg-verydeep-19.mat' cnn_feat
                load '/home/chuna/Zeroshot/SSE/cnn-features/CUB_200_2011/class_attribute_labels_continuous.mat' classAttributes
                load '/home/chuna/Zeroshot/SSE/cnn-features/CUB_200_2011/image_class_labels.mat'  imageClassLabels
                load '/home/chuna/Zeroshot/SSE/cnn-features/CUB_200_2011/train_test_split.mat' train_cid test_cid            
                X_all = normc(cnn_feat);
                Y_all = double(imageClassLabels(:,2)');
                non_zero_category = train_cid;
                zero_category = test_cid;
                attr_table = normc(double(classAttributes)');
            elseif(strcmp(feature,'google_net'))
                load '/home/chuna/Zeroshot/SynC/data/CUB_googlenet' X;
                load '/home/chuna/Zeroshot/SynC/data/CUB_inform_release' attr2 Y CUB_class_split;
                X_all = normc(X');
                Y_all = Y;
                zero_category = CUB_class_split(1,:);
                non_zero_category = setdiff(1:200,zero_category);
                attr_table = normc(double(attr2)');
            end
            
        end
        %{
        class_min_num = min(hist(Y_all ,unique(Y_all)));
        occur_num = floor(class_min_num*0.8);
        
        train_id = [];
        test_id = [];
        for id = unique(Y_all)
            class_occur_id = find(Y_all == id);
            id_list = randperm(length(class_occur_id));
            if(ismember(id, non_zero_category ))
                train_id = [train_id class_occur_id(id_list(1:occur_num))];
            else
                test_id = [test_id class_occur_id(id_list(1:occur_num))];
            end
            test_id = [test_id  class_occur_id(id_list(occur_num+1:class_min_num))];
        end   
        %}
        y_list = unique(Y_all);
        class_num = hist(Y_all ,y_list);
        occur_num = floor(class_num*0.8);
        
        train_id = [];
        test_id = [];
        for id = 1:length(y_list)
            yid = y_list(id);
            class_occur_id = find(Y_all == yid);
            id_list = randperm(length(class_occur_id));
            if(ismember(id, non_zero_category ))
                train_id = [train_id class_occur_id(id_list(1:occur_num(id)))];
            else
                test_id = [test_id class_occur_id(id_list(1:occur_num(id)))];
            end
            test_id = [test_id  class_occur_id(id_list(occur_num(id)+1:end))];
        end   
        
        data.Xs = X_all(:,train_id);                               
        data.Ys = Y_all(:,train_id);
        data.Xl = attr_table(:,non_zero_category); 
        data.Yl = non_zero_category;
        data.Xlh = attr_table(:,zero_category); 
        data.Ylh = zero_category;
        data.Xsh = X_all(:,test_id);
        data.Ysh = Y_all(:,test_id);
        acc_all(k,i) = TransferNeuralTreesZSLmixtest(data);     
    end
end