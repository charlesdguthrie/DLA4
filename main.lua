--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

-- TODOS:
-- save model to disk
-- write inverse dictionary
-- rewrite communication loop call run_test() on saved model. 
-- run_test() will return pred, among other things
-- communication loop will run pred through inverse dictionary to produce predicted word.


gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

require('nngraph')
require('base')
ptb = require('data')

-- Trains 1 epoch and gives validation set ~182 perplexity (CPU).
local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length: number of blocks
                layers=2, -- how many LSTM stacks
                decay=2,
                rnn_size=200, -- hidden unit size.  Size of vector input
                dropout=0, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate (default 4)
                max_max_epoch=13, -- final epoch (default 13)
                max_grad_norm=5, -- clip when gradients exceed this norm value.  TODO: modify for gradient clipping
                model_path = 'model.net',
                best_model_path = 'best_model.net',
                vocab_map_path = 'vocab_map.tab',
                save_freq = 1, --save model every n epochs
                patience = 3,
                rnn_type = 'lstm' -- 'lstm' or 'gru'
               }

function transfer_data(x)
    -- not used. We are not using GPUs.
    if gpu then
        return x:cuda()
    else
        return x
    end
end

--initialize model global param
model = {}

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x) --input to hidden
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h) -- hidden to hidden
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

local function grucell(x, prev_c, prev_h)
    -- this will be Wr, Wu, Whtilda
    local i2h   = nn.Linear(params.rnn_size, 3*params.rnn_size)(x)
    -- Ur, Uu, Uhtilda
    local h2h   = nn.Linear(params.rnn_size, 3*params.rnn_size)(prev_h)
    -- take first 2 (Wr, Wu)
    local gates = nn.CAddTable()({
            nn.Narrow(2, 1, 2 * params.rnn_size)(i2h),
            nn.Narrow(2, 1, 2 * params.rnn_size)(h2h),
    })
    -- split into Wr and Wu
    gates = nn.SplitTable(2)(nn.Reshape(2, params.rnn_size)(gates))
    -- r
    local resetgate  = nn.Sigmoid()(nn.SelectTable(1)(gates))
    -- u
    local updategate = nn.Sigmoid()(nn.SelectTable(2)(gates))
    -- htilda
    local output = nn.Tanh()(nn.CAddTable()({
            nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(i2h),
            nn.CMulTable()({resetgate,
                nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(h2h),})
    }))
    local next_h = nn.CAddTable()({ prev_h,
            nn.CMulTable()({ updategate,
                nn.CSubTable()({output, prev_h,}),})
    })

    --just pass c through
    local next_c = prev_c
    return next_c, next_h
end

function create_network() --TODO: add a parameter that allows you to swap out lstm with gru
    -- creates single unit of network
    local x                  = nn.Identity()() -- current word
    local y                  = nn.Identity()() -- actual next word
    local prev_s             = nn.Identity()() -- prev state: concat (prev_c, prev_h)
    --define zeroth element.  Lookup table allows you to get a rnn_size vector embedding of each word
    local i                  = {[0] = nn.LookupTable(params.vocab_size, 
                                                    params.rnn_size)(x)} 
    local next_s             = {}
    local split              = {prev_s:split(2 * params.layers)} -- contains c and h
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1] -- pull out c
        local prev_h         = split[2 * layer_idx] -- pull out h
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1]) --implement dropout

        --select lstm or gru
        if params.rnn_type == 'lstm' then
            local next_c, next_h = lstm(dropped, prev_c, prev_h)
        elseif params.rnn_type == 'gru' then
            local next_c, next_h = grucell(dropped, prev_c, prev_h)
        end
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped))
    local err                = nn.ClassNLLCriterion()({pred, y})
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), pred}) --TODO: output a third thing: pred.  It won't break Lua to output 3 things when it asks for 2. NOTE: I already did it.  if it breaks, the error is here.  NOTE2: this returns log(pred) but ranking is the same.
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

function setup()
    -- builds RNN network with initial states
    print("Creating a RNN LSTM network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {} -- state at time step
    model.ds = {} -- param.layer
    model.start_s = {}
    for j = 0, params.seq_length do --j means timestep
        model.s[j] = {}
        for d = 1, 2 * params.layers do -- d is layer index, h/c
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
    -- zeros out everything in state
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- given state, calculate error.
    -- g_replace_table(from, to).  
    g_replace_table(model.s[0], model.start_s)
    
    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end

    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i], pred = unpack(model.rnns[i]:forward({x, y, s})) --NOTE: added pred
        state.pos = state.pos + 1
    end
    
    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])
    
    -- cross entropy error
    return model.err:mean()
end

function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1?
        -- backward pass requires a value for dpred, like derr
        local derr = transfer_data(torch.ones(1))
        local dpred = transfer_data(torch.zeros(params.batch_size,params.vocab_size)) --TODO: should it be batch_size?
        -- tmp stores the ds
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds, dpred})[3]
        -- remember (to, from)
        g_replace_table(model.ds, tmp)
    end
    
    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length
    
    -- gradient clipping TODO: modify this if youlike
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    
    -- gradient descent step
    paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)
    
    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)
    
    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    valid_perp = torch.exp(perp / len)

    -- save current model every n epochs
    if epoch % params.save_freq == 0 then
        print('saving model to '..params.model_path)
        torch.save(params.model_path,model)
    end

    -- save best model if current perp is better than best perp
    if best_valid_perp then
        if valid_perp < best_valid_perp then
            best_valid_perp = valid_perp
            wait = 0
            print('saving best model to '..params.best_model_path)
            torch.save(params.best_model_path,model)
        else -- otherwise wait.  Once wait > patience, give up.
            wait = wait + 1
        end
    else best_valid_perp = valid_perp
    end

    print("Validation set perplexity : " .. g_f3(valid_perp))
    g_enable_dropout(model.rnns)
end

function run_test()
    -- makes one jump through
    reset_state(state_test)
    g_disable_dropout(model.rnns) --bc you're in test mode
    local perp = 0 --perplexity
    local len = state_test.data:size(1) --num data points in test set
    
    -- no batching here
    -- copy (to, from).  start_s is the first hidden state before any words.  start_s will be all zeros
    g_replace_table(model.s[0], model.start_s) 
    for i = 1, (len - 1) do -- use each word to predict the next, up to using 19 to predict 20
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]})) -- recall that model.s is a memory of what you've already said. NOTE: added pred
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end


if gpu then
    g_init_gpu(arg)
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

-- save vocab map
print("Saving vocab map to "..params.vocab_map_path)
torch.save(params.vocab_map_path,ptb.vocab_map)

print("Network parameters:")
print(params)

local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
    reset_state(state)
end
setup()
step = 0
epoch = 0
total_cases = 0
wait = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

while epoch < params.max_max_epoch do

    -- take one step forward
    perp = fp(state_train)
    if perps == nil then
        perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    
    -- gradient over the step
    bp(state_train)
    
    -- words_per_step covered in one step
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    
    -- display details at some interval
    if step % torch.round(epoch_size / 10) == 10 then
        wps = torch.floor(total_cases / torch.toc(start_time)) -- words per second
        since_beginning = g_d(torch.toc(beginning_time) / 60)
        print('epoch = ' .. g_f3(epoch) ..
             ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
             ', wps = ' .. wps ..
             ', dw:norm() = ' .. g_f3(model.norm_dw) ..
             ', lr = ' ..  g_f3(params.lr) ..
             ', since beginning = ' .. since_beginning .. ' mins.')
    end
    -- TODO: save model periodically.
    -- run when epoch done
    if step % epoch_size == 0 then
        run_valid()
        if epoch > params.max_epoch then
            params.lr = params.lr / params.decay
        end
    end

    -- stop early if wait exceeds patience (ie the model has stopped improving)
    if wait >= params.patience then
        print("early stop. " .. torch.floor(epoch) .. " epochs")
        break
    end
end
run_test()
print("Training is over.")
