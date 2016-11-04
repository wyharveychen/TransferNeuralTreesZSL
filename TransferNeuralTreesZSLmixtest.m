function acc_test = TransferNeuralTreesZSLmixtest(data,prev_model)
    addpath novelty_detection/
    %% General Initialization
    split_method_list = {'cond','thres','gaussian','loop','stack'};
    %seen.(data,data_label,attr,attr_label)
    %unseen.(data,data_label,attr,attr_label)
    data_dim            = size(data.Xs,1);    
    attr_dim            = size(data.Xl,1);
    class_num           = length(data.Yl);
    common_dim          = 100;
    %seen_out_dim        = length(seen_label_list); %only used when directly project S to T
    thres_list     = [-1000,-100,-10:1:-2,-1.2:0.001:-1,-1:0.05:0];   
    
    %% New variables for clearity
    seen_label_list     = unique(data.Yl);
    unseen_label_list   = unique(data.Ylh);    
    test_seen_id        = ismember(data.Ysh, seen_label_list);
    test_unseen_id      = ismember(data.Ysh, unseen_label_list);
    
    %% New variables for recording
    acc = zeros(1,5); %acc of training seen data, training seen attr, training unseen attr, test seen data, test unseen data
    emloss_train = 0; emloss_test = 0; emloss_test_seen = 0; emloss_test_unseen = 0;
    t_acc.thres     = thres_list;
    t_acc.seen      = zeros(size(thres_list));
    t_acc.unseen    = zeros(size(thres_list));    
    t_acc.all       = zeros(size(thres_list));
    t_acc.soru_s    = zeros(size(thres_list));
    t_acc.soru_u    = zeros(size(thres_list)); 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Start Training and test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Initialize seen routes
    data_mapping        = BasicLayers(data_dim,common_dim);
    %data_mapping1        = BasicLayers(data_dim,2*common_dim);
    %norm_mapping         = L2NormLayers();
    %data_mapping2        = BasicLayers(2*common_dim,common_dim);


    attr_mapping        = BasicLayers(attr_dim,common_dim);  
    seen_classifier     = NDFLayers(common_dim,seen_label_list);
            
    seen_data_path = NN({data_mapping,seen_classifier});
    %seen_data_path = NN({data_mapping1,norm_mapping,data_mapping2,seen_classifier});
    
    seen_attr_path = NN({attr_mapping,seen_classifier});
   
    %% Train with seen class data
    seen_data_path.Train(data.Xs,data.Ys,struct('epoch_num',200,'batch_num',5,'converge_acc',1));       
    acc(1) = mean(seen_data_path.Predict(data.Xs) == data.Ys);        
    fprintf('Source train acc = %d\n',acc(1));
    seen_classifier.RecordForestNorm(data.Ys);
    emloss_train = seen_classifier.EmbeddingLoss(); 
    mapped_train_data = data_mapping.next_in; %record
    %seen_data = data_mapping.next_in; %record
    %data_cost = seen_data_path.final_cost;
  
    %% Train with seen class attriute
    seen_attr_path.ClearCache();
    seen_attr_path.Train(data.Xl,data.Yl,struct('epoch_num',500,'updatestop_layer',1));       
    acc(2) = mean(seen_attr_path.Predict(data.Xl) == data.Yl);            
    fprintf('Target train acc = %d\n',acc(2));
    mapped_seen_attr = attr_mapping.next_in; %record
    %seen_attr = attr_mapping.next_in;
    %attr_cost = seen_attr_path.final_cost;
            
    %% Initialize unseen routes (using seen routes)
    unseen_classifier = copy(seen_classifier);
    unseen_data_path = NN({data_mapping,unseen_classifier});
    %unseen_data_path = NN({data_mapping1,norm_mapping,data_mapping2,unseen_classifier});
    
    unseen_attr_path = NN({attr_mapping,unseen_classifier});    
    
    %% Update classifier with unseen attribute
    unseen_attr_path.UpdateUnseenClass(data.Xlh,data.Ylh);           
    acc(3) = mean(unseen_attr_path.Predict(data.Xlh) == data.Ylh);                    
    fprintf('Target test acc = %d\n',acc(3));
    %unseen_attr = attr_mapping.next_in;
    
    %% Test seen class accuracy
    predicted_seen_Ysh = seen_data_path.Predict(data.Xsh);   
    predicted_seen_prob = seen_classifier.next_in;
    
    acc(4) = mean(predicted_seen_Ysh(:,test_seen_id ) == data.Ysh(:,test_seen_id));        
    fprintf('Source seen test acc = %d\n',acc(4));    
    
    %% Test unseen class accuracy
    predicted_unseen_Ysh = unseen_data_path.Predict(data.Xsh);    
    predicted_unseen_prob = unseen_classifier.next_in;

    acc(5) = mean(predicted_unseen_Ysh(:,test_unseen_id ) == data.Ysh(:,test_unseen_id));                    
    fprintf('Source unseen test acc = %d\n',acc(5));    
    %unseen_data = data_mapping.next_in;
    mapped_test_data = data_mapping.next_in; 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Test under different s/u thres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            
    %% mix prediction under different threshold (bias probability)
    seen_data_path.Predict(data.Xsh);   
    emloss_test = seen_classifier.EmbeddingLoss(); 
    emloss_test_seen = emloss_test(:,test_seen_id);
    emloss_test_unseen = emloss_test(:,test_unseen_id);
%{
    unseen_data_path.Predict(data.Xs);        
    emloss_train2 = unseen_classifier.EmbeddingLoss(); 
    unseen_data_path.Predict(data.Xsh);        
    emloss_test2 = unseen_classifier.EmbeddingLoss(); 
    emloss_test_seen2 = emloss_test2(:,test_seen_id);
    emloss_test_unseen2 = emloss_test2(:,test_unseen_id);
%}    

    [mu, sigma, priors] = trainMultiVariantGaussianDiscriminant(mapped_train_data, data.Ys, mapped_seen_attr, data.Yl);
    logprobabilities = predictMultiVariantGaussianDiscriminant(mapped_test_data, mu, sigma, priors);

    lambda = 7;
    kNN = 20;
    loop_prob = calcOutlierPriors( mapped_test_data, mapped_train_data, data.Ys, data.Yl, lambda, kNN);

    for split_id = 1:4
        split_method =  split_method_list{split_id};
        for thres_id = 1:length(thres_list)          
            thres =  thres_list(thres_id); 
            predicted_Ysh = predicted_seen_Ysh;        
            if(strcmp(split_method,'cond'))
                %p_seen = tanh(thres*emloss_test);
                p_seen = 2./(1+exp(-thres*emloss_test/class_num ))-1;
                [~,max_id] = max([bsxfun(@times,predicted_seen_prob,p_seen);bsxfun(@times,predicted_unseen_prob,(1-p_seen))],[],1);                     
                unseen_predict_id = max_id>length(seen_classifier.label_list);
                predicted_Ysh(:,unseen_predict_id) = predicted_unseen_Ysh(:, unseen_predict_id );
                t_acc.soru_s(thres_id) = mean(p_seen(:,test_seen_id)>=0.5);
                t_acc.soru_u(thres_id) = mean(p_seen(:,test_unseen_id)<0.5);
            elseif(strcmp(split_method,'stack'))
                [~,max_id] = max([predicted_seen_prob+thres;predicted_unseen_prob],[],1);                     
                unseen_predict_id = max_id>length(seen_classifier.label_list);
                predicted_Ysh(:,unseen_predict_id) = predicted_unseen_Ysh(:, unseen_predict_id );
            elseif(strcmp(split_method,'thres'))
                predicted_Ysh(:,emloss_test>class_num*thres) = predicted_unseen_Ysh(:,emloss_test>class_num *thres);
                t_acc.soru_s(thres_id) = mean(emloss_test(:,test_seen_id)<=class_num*thres);
                t_acc.soru_u(thres_id) = mean(emloss_test(:,test_unseen_id)>class_num*thres);
            elseif(strcmp(split_method,'gaussian'))
                %max_seen_prob = max(predicted_seen_prob,[],1);                                     
                predicted_Ysh(:,logprobabilities>3*thres) = predicted_unseen_Ysh(:,logprobabilities>3*thres);
                %range -1 ~ -3
                t_acc.soru_s(thres_id) = mean(logprobabilities(:,test_seen_id)<=3*thres);
                t_acc.soru_u(thres_id) = mean(logprobabilities(:,test_unseen_id)>3*thres);
            elseif(strcmp(split_method,'loop'))    
                predicted_Ysh(:,loop_prob>= -thres) = predicted_unseen_Ysh(:,loop_prob>= -thres);                
                t_acc.soru_s(thres_id) = mean(loop_prob(:,test_seen_id)< -thres);
                t_acc.soru_u(thres_id) = mean(loop_prob(:,test_unseen_id)>= -thres);
            end

            t_acc.seen(thres_id) = mean(predicted_Ysh(:,test_seen_id) == data.Ysh(:,test_seen_id));        
            fprintf('Source seen test full class prediction acc = %d\n',t_acc.seen(thres_id));    

            t_acc.unseen(thres_id) = mean(predicted_Ysh(:,test_unseen_id) == data.Ysh(:,test_unseen_id));    
            fprintf('Source unseen test full class prediction acc = %d\n',t_acc.unseen(thres_id));    

            t_acc.all(thres_id) = mean(predicted_Ysh == data.Ysh);
        end
        t_acc_list{split_id} = t_acc;
    end
    split_method_used = split_method_list(1:4);
    %{
    figure(1);
    plot(thres_list,t_acc.unseen);
    hold on;
    plot(thres_list,t_acc.seen);
    hold off;
    text(-1,acc_overall(6),sprintf('overall acc:%f\n', acc_overall(6)));
    %}
    plot(t_acc.soru_u,t_acc.soru_s);
    plot(t_acc.unseen,t_acc.seen);   
    t = clock;
    data_name = sprintf('./record_data/TNTZSL_%s_acc_all_%04d%02d%02d%02d%02d_mixtest',data.dataset,t(1),t(2),t(3),t(4),t(5));
    save(data_name,'t_acc','t_acc_list','split_method_used','acc','emloss_test_seen','emloss_test_unseen','emloss_train');
    data_name =  sprintf('./record_data/TNTZSL_%s_acc_mixtest',data.dataset);
    %save(data_name,'l_acc','g_acc','t_acc');
    save(data_name,'t_acc','t_acc_list','split_method_used','acc','emloss_test_seen','emloss_test_unseen','emloss_train');
    %data_name =  sprintf('./record_data/TNTZSL_%s_acc_mixtest_unseenroute',data.dataset);
    %save(data_name,'l_acc','g_acc','t_acc');
    %save(data_name,'emloss_test_seen2','emloss_test_unseen2','emloss_train2');
    
    %{ 
    %CMT
    load(sprintf('/home/chuna/Zeroshot/zslearning-master/%s_acc.mat', data.dataset),'g_acc','gausian_results','l_acc','loop_results');   
    figure(2);
    plot(l_acc.thres,l_acc.unseen);
    hold on;
    plot(l_acc.thres,l_acc.seen);
    hold off;
    text(0,loop_results.accuracy,sprintf('overall acc:%f\n',loop_results.accuracy));

    figure(3);
    plot(1:11,g_acc.unseen);
    hold on;
    plot(1:11,g_acc.seen);
    hold off;
    text(0,gausian_results.accuracy,sprintf('overall acc:%f\n',gausian_results.accuracy));
    
    figure(4);
    plot(acc_seen,acc_unseen);
    hold on;
    plot(l_acc.seen,l_acc.unseen);
    plot(g_acc.seen,g_acc.unseen);
    hold off;
    %}
   
    %{
    seen_label = data.Ys;
    seen_attr_class = data.Yl;
    unseen_label = data.Ysh(:,test_unseen_id);
    unseen_attr_class = data.Ylh;
    %}
    
    %mean(var_seen<thres)
    %mean(var_unseen>thres)
       
    %t = clock;
    %data_name = sprintf('./record_data/TNTZSL_%s_data_%04d%02d%02d%02d%02d',data.dataset,t(1),t(2),t(3),t(4),t(5));
    %save(data_name,'seen_data','seen_label','seen_attr','seen_attr_class','unseen_data','unseen_label','unseen_attr','unseen_attr_class','acc','data_cost','attr_cost','var_seen','var_unseen');
    
    %model_name = sprintf('./record_model/TNTZSL_%s_model_%04d%02d%02d%02d%02d',data.dataset,t(1),t(2),t(3),t(4),t(5));    
    %save(model_name,'seen_data_path','seen_attr_path','unseen_data_path','unseen_attr_path','acc');
    
    acc_test = acc(5); %return
end
function SaveModel(dataset,data_path,attr_path,acc)
    %data_path.ClearCache();
    save(model_name,'data_path','attr_path','acc');
end