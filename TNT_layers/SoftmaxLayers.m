classdef SoftmaxLayers <handle
    properties  
        type = 'Softmax';
        weights; %[pre,]
        is_final_layer = 1;
        
        %cache parameter of updating
        prob;
        y_embd;
        
        label_list;
        label_map;
    end
    
    methods
        function layer = SoftmaxLayers(label_list)
             layer.label_list = label_list; 
             layer.label_map = containers.Map(label_list,1:length(label_list));                 
        end
        function next_in = Forward(layer,in) %in: input, next_in: layer output (next layer input)       
             next_in = layer.SoftMax(in);
             layer.prob = next_in;
        end
        function prev_derr = Backward(layer,derr) %derr: derr/ds
             layer.y_embd = layer.LabelEmbedding(derr);
             prev_derr = (layer.prob - layer.y_embd);             
        end
        function Update(layer)               
        end
        
        function out = LabelEmbedding(layer,y)
             yid = cell2mat(values(layer.label_map,mat2cell(y', ones(size(y)))))';
             out = full(sparse(yid,1:length(yid),ones(length(yid),1)));
        end
        function cost = TotalError(layer)                                   
            cost = 0.5*mean(sum((layer.y_embd - layer.prob).^2,1));
        end
        function predicted_label = Predict(layer)
            [~,max_id] = max(layer.prob,[],1);
            predicted_label = layer.label_list(max_id);
        end
        function ClearCache(layer)
            layer.prob = [];
            layer.y_embd = [];
        end

    end
    methods(Static)
        function y = SoftMax(x)
            z = exp(bsxfun(@minus, x, max(x)));
            y = bsxfun(@rdivide, z, sum(z));
        end      
    end
end