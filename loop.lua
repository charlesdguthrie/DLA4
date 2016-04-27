-- loop.lua
-- loops through parameters for experimentation
--[[
Checklist:
1. modify var_list
2. set varied parameter equal to var
3. put varied parameter in model name
]]


-- list of parameters over which to loop
var_list = {100,200,400,600}

for i,var in ipairs(var_list) do

	params = {
	        batch_size=40, -- minibatch
	        seq_length=20, -- unroll length: number of blocks
	        layers=2, -- how many LSTM stacks
	        decay=2,
	        rnn_size=var, -- hidden unit size.  Size of vector input
	        dropout=0.1, 
	        init_weight=0.1, -- random weight initialization limits
	        lr=1, --learning rate
	        vocab_size=10000, -- limit on the vocabulary size
	        max_epoch=4,  -- when to start decaying learning rate (default 4)
	        max_max_epoch=25, -- final epoch (default 13)
	        max_grad_norm=3, -- clip when gradients exceed this norm value.  TODO: modify for gradient clipping
	        model_name = 'model_20160426_size_'..var,
	        vocab_map_path = 'vocab_map.tab',
	        save_freq = 1, --save model every n epochs
	        patience = 3,
	        rnn_type = 'lstm', -- 'lstm' or 'gru'
            debug = false
	       }
    print("---------------------------------- \n -- RUNNING MODEL ".. params.model_name .." --\n------------------------------------")
	dofile('main.lua')
end
