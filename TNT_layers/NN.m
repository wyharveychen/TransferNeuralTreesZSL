classdef NN < handle
   properties       
       layers;
       layer_num;
       inputs;
       derrs;
       final_cost;
       is_iteratively_update = 0;       
   end
   methods
        function nn = NN(layers)
            nn.layers = layers;
            nn.layer_num =  length(nn.layers);

        end
        function Train(nn,train_data,train_label,opt)  
            batch_num        = NN.Initialize(opt,'batch_num',1);
			epoch_num        = NN.Initialize(opt,'epoch_num', 50);
            input_layer      = NN.Initialize(opt,'input_layer',1);
            output_layer     = NN.Initialize(opt,'output_layer',nn.layer_num);
            updatestop_layer = NN.Initialize(opt,'updatestop_layer',output_layer);
            converge_acc     = NN.Initialize(opt,'converge_acc',1.01);% default: no converge

                      
            batch_size = floor(size(train_data, 2)/ batch_num);

            ndf_pi_allbatch = cell(1,batch_num);            
            if(nn.is_iteratively_update && strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                ndf_pi_allbatch_treewise = cell(nn.layers{nn.layer_num}.tree_num,batch_num);            
            end
            for epoch = 1:epoch_num
                batch_perm_id = reshape(randperm(size(train_data, 2), batch_num * batch_size),batch_num,batch_size);                
                total_cost = 0;
                train_acc = 0;
                for batch = 1:batch_num                    
                    nn.inputs{1} =  train_data(:,batch_perm_id(batch,:));
                    nn.derrs{nn.layer_num+1} = train_label(batch_perm_id(batch,:));
                                        
                    if(nn.is_iteratively_update && strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                        for tree_id = 1:nn.layers{nn.layer_num}.tree_num
                            for l_id = input_layer:output_layer 
                                if(strcmp(nn.layers{l_id}.type, 'NDF') )
                                    nn.inputs{l_id+1} = nn.layers{l_id}.ForwardOneTree(nn.inputs{l_id},tree_id);
                                else
                                    nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
                                end
                            end                            
                            for l_id = output_layer:-1:input_layer
                                if(strcmp(nn.layers{l_id}.type, 'NDF') )
                                    nn.derrs{l_id} = nn.layers{l_id}.BackwardOneTree(nn.derrs{l_id+1},tree_id);
                                else
                                    nn.derrs{l_id} = nn.layers{l_id}.Backward(nn.derrs{l_id+1});
                                end
                            end
                            for l_id = input_layer:updatestop_layer
                                if(strcmp(nn.layers{l_id}.type, 'NDF') )
                                    nn.layers{l_id}.UpdateOneTree(tree_id);
                                else
                                    nn.layers{l_id}.Update();
                                end
                            end
                            if(updatestop_layer>=nn.layer_num)
                                ndf_pi_allbatch_treewise{tree_id,batch} = nn.layers{nn.layer_num}.pi{tree_id};     
                                average_pi = ndf_pi_allbatch_treewise{tree_id,1};                        
                                for i = 2:batch_num
                                    if(isempty(ndf_pi_allbatch_treewise{tree_id,i}))
                                        i = i-1;
                                        break;
                                    end
                                    average_pi = average_pi + ndf_pi_allbatch_treewise{tree_id,i};
                                end
                                average_pi = average_pi/i; 
                                nn.layers{nn.layer_num}.UpdatePIDirectlyOneTree(average_pi,tree_id);                                
                            end
                        end                        
                    else                     
                        for l_id = input_layer:output_layer 
                            nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
                        end
                        for l_id = output_layer:-1:input_layer
                            nn.derrs{l_id} = nn.layers{l_id}.Backward(nn.derrs{l_id+1});
                        end
                        for l_id = input_layer:updatestop_layer
                            nn.layers{l_id}.Update();
                        end                                        
                        %total_cost = total_cost+ nn.layers{nn.layer_num}.TotalError();
                        %% Fully update PI                   
                        if( strcmp(nn.layers{nn.layer_num}.type, 'NDF') && updatestop_layer>=nn.layer_num)
                            ndf_pi_allbatch{batch} = nn.layers{nn.layer_num}.pi;     
                            average_pi = ndf_pi_allbatch{1};                        
                            if(batch_num ~= 1)
                                for i = 2:batch_num
                                    if(isempty(ndf_pi_allbatch{i}))
                                        i = i-1;
                                        break;
                                    end
                                    average_pi = cellfun(@plus,average_pi,  ndf_pi_allbatch{i},'UniformOutput',0);
                                end                           
                                average_pi = cellfun(@(pi) pi/i,average_pi,'UniformOutput',0);
                            end
                            nn.layers{nn.layer_num}.UpdatePIDirectly(average_pi);
                        end
                    end
                    total_cost = total_cost+ nn.layers{nn.layer_num}.TotalError();                    
                    train_acc = train_acc+ mean(nn.layers{nn.layer_num}.Predict() == nn.derrs{nn.layer_num+1});
                end
                nn.final_cost = total_cost;               
                %predicted_label = Predict(nn,train_data);
                train_acc = train_acc/batch_num;
                fprintf('Epoch %d, Training Cost: %d, Train Acc: %d\n',epoch, total_cost, train_acc);
                if(converge_acc<=train_acc)
                    break;
                end
            end
        end
        function predicted_label = Predict(nn,test_data)
            nn.inputs{1} =  test_data;
            for l_id = 1:nn.layer_num
                nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
            end
            predicted_label = nn.layers{nn.layer_num}.Predict();
        end
        function ClearCache(nn)
            for l_id = 1:nn.layer_num
                nn.layers{l_id}.ClearCache();
            end
        end
        function UpdateUnseenClass(nn,in,y)
            nn.inputs{1} =  in;
            nn.layers{nn.layer_num}.InitializePILayer(unique(y));
            for l_id = 1:nn.layer_num
                nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
            end
            %nn.layers{nn.layer_num}.Backward(y);
                %nn.layers{nn.layer_num}.UpdatePIOnly(y);            
            nn.layers{nn.layer_num}.UpdatePIOnly(y,1/100/length(unique(y)));                        
        end
   end
   methods(Static)
       function value = Initialize(arg,field,default)
            if(isfield(arg,field))
                value = eval(sprintf('arg.%s;',field));
            else
                value = default;
            end            
        end
   end
end