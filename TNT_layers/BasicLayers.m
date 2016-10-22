classdef BasicLayers <handle
    properties  
        %
        type = 'Basic';
        weights; %[pre,]
        BasicNeuron;
        BasicNeuronDiff;         
        
        is_final_layer = 0;

        %cache parameter of updating
        in;
        next_in;
        grad;   
        
        G;
        D;
        m;
        
        %for optimization
        learn_rate;
        learn_rate_Adam = 5e-2; 
        learn_rate_RMSprop = 0.5e-2; 
        learn_rate_adadelta = 5; 
        learn_rate_adagrad = 5e-2; 
        learn_rate_noadagrad = 1; 
        ada_decay_rate = 0.9;
    
        optimizer;
    end
    
    methods
        function layer = BasicLayers(indim,outdim)
            layer.weights = [(rand(indim, outdim) - 0.5); zeros(1,outdim)]; 
            layer.G = zeros(size(layer.weights));
            layer.D = zeros(size(layer.weights));
            layer.m = zeros(size(layer.weights));

            layer.BasicNeuron = @BasicLayers.Tanh;
            layer.BasicNeuronDiff = @BasicLayers.TanhDiff;  
             
            layer.SetOpimizer('RMSprop');     
        end                
        function SetOpimizer(layer,optimizer)
            layer.optimizer = optimizer;
            if(strcmp(layer.optimizer,'Adam'))
                layer.learn_rate = layer.learn_rate_Adam;
            elseif(strcmp(layer.optimizer,'RMSprop'))
                layer.learn_rate = layer.learn_rate_RMSprop;
            elseif(strcmp(layer.optimizer,'adadelta'))
                layer.learn_rate = layer.learn_rate_adadelta;
            elseif(strcmp(layer.optimizer,'adagrad'))
                layer.learn_rate = layer.learn_rate_adagrad;
            else    
                layer.learn_rate = layer.learn_rate_noadagrad;           
            end
        end        
        function next_in = Forward(layer,in) %in: input, next_in: layer output (next layer input)
             in_num = size(in,2);
             layer.in = [in;ones(1, in_num)]; 
             next_in = layer.BasicNeuron(layer.weights' * layer.in);
             layer.next_in = next_in;
        end
        function prev_derr = Backward(layer,derr) %derr: derr/ds
             in_num = size(derr,2);
             %prev_derr = (layer.weights * derr).* layer.BasicNeuronDiff(layer.in);
             prev_derr = (layer.weights * (derr.* layer.BasicNeuronDiff(layer.next_in)) ); 
             %derr is prev derr/ds *w, before become derr/s in this layer, first multiply by theta' 
             layer.grad = (layer.in*(derr.* layer.BasicNeuronDiff(layer.next_in))')/in_num;
             prev_derr = prev_derr(1:end-1,:);
        end
        function Update(layer)
            if(strcmp(layer.optimizer,'Adam'))
                adam_beta1 = 0.9;
                adam_beta2 = 0.999;
                layer.G = adam_beta2*layer.G + (1-adam_beta2)*layer.grad.^2;
                layer.m = adam_beta1*layer.m + (1-adam_beta1)*layer.grad;
                layer.weights = layer.weights - layer.learn_rate *(layer.m/(1-adam_beta1))./(sqrt(layer.G/(1-adam_beta2))+1e-8);    
            elseif(strcmp(layer.optimizer,'RMSprop'))
                layer.G = layer.ada_decay_rate*layer.G + (1-layer.ada_decay_rate)*layer.grad.^2;
                layer.weights = layer.weights - layer.learn_rate *layer.grad./(sqrt(layer.G)+1e-6);    
            elseif(strcmp(layer.optimizer,'adadelta'))
                layer.G = layer.ada_decay_rate*layer.G + (1-layer.ada_decay_rate)*layer.grad.^2;
                x = -sqrt(layer.D+0.1)./sqrt(layer.G+0.1).*layer.grad;
                layer.D = layer.ada_decay_rate*layer.D + (1-layer.ada_decay_rate)*x.^2;
                layer.weights = layer.weights + layer.learn_rate*x;    
            elseif(strcmp(layer.optimizer,'adagrad'))
                layer.G = layer.G + layer.grad.^2;
                layer.weights = layer.weights - layer.learn_rate *layer.grad./(sqrt(layer.G)+1e-6);    
             else
                layer.weights = layer.weights - layer.learn_rate *layer.grad;             
             end
        end
        function ClearCache(layer)
            layer.in = [];
            layer.next_in = [];
            layer.grad = [];   
        end
    end
    methods(Static)   
        %% for math
        function y = Tanh(x)
            y = tanh(x);
        end
        function y = TanhDiff(x)
            y = (1-x.^2);
        end      
        function y = ReLU(x)
			y = max(x, 0);
		end
		function y = diffReLU(z)
			y = z > 0;
        end
        function y = Linear(x)
			y = x;
		end
		function y = diffLinear(z)
			y = 1;
        end
        function y = Bent(x)            
			y = x.*(x>0) + 0.1*x.*(x<0);
		end
		function y = diffBent(z)
			y = (z>0) + 0.1*(z<0) + 0.55*(z==0);
        end
    end
end