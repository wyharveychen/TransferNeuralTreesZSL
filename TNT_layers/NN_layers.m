classdef NN_layers < handle
    properties  
        weights; %[pre,]
        parameters;
        isFinalLayer;
    end
    
    methods
        function next_in = Forward(in)
        end
        function [prev_out,grad] = Backward(out)
        end
        function Update(grad)        
        end
    end
    
end