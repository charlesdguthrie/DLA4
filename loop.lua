-- loop.lua
-- loops through parameters for experimentation
--[[
Checklist:
1. modify var_list
2. set varied parameter equal to var
3. put varied parameter in model name
]]


-- list of parameters over which to loop
var_list = {1,2,3}

for i,var in ipairs(var_list) do
	params = {
	        batch_size=20, -- minibatch
	        seq_length=20, -- unroll length: number of blocks
	        layers=var, -- how many LSTM stacks
	        decay=2,
	        rnn_size=200, -- hidden unit size.  Size of vector input
	        dropout=0, 
	        init_weight=0.1, -- random weight initialization limits
	        lr=1, --learning rate
	        vocab_size=10000, -- limit on the vocabulary size
	        max_epoch=4,  -- when to start decaying learning rate (default 4)
	        max_max_epoch=13, -- final epoch (default 13)
	        max_grad_norm=5, -- clip when gradients exceed this norm value.  TODO: modify for gradient clipping
	        model_name = 'model_20160425_layers_'..var,
	        vocab_map_path = 'vocab_map.tab',
	        save_freq = 1, --save model every n epochs
	        patience = 3,
	        rnn_type = 'lstm' -- 'lstm' or 'gru'
	       }
	dofile('main.lua')
end