--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

-- define file names
if params.debug == true then
    trainfn = ptb_path .. "ptb.train_head.txt"
    testfn  = ptb_path .. "ptb.test_head.txt"
    validfn = ptb_path .. "ptb.valid_head.txt"
else
    trainfn = ptb_path .. "ptb.train.txt"
    testfn  = ptb_path .. "ptb.test.txt"
    validfn = ptb_path .. "ptb.valid.txt"
end

local vocab_idx = 0
local vocab_map = {}
local inverse_map = {}

-- start with all zeros, then fill in values
-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1 --1st cut point in current spot in loop
        local finish = start + x:size(1) - 1 -- 2nd cut point in current loop
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

-- CG: Loads data, replacing new lines with eos, splitting sentences into words
-- CG: also builds dictionary
local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>') --identify eos
    data = stringx.split(data) -- split sentences into words
    print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then -- build the dictionary
            vocab_idx = vocab_idx + 1 -- 
            vocab_map[data[i]] = vocab_idx
        end
        x[i] = vocab_map[data[i]] -- x contains a pointer to dictionary id
    end
    return x
end

local function traindataset(batch_size, char)
   local x = load_data(trainfn)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
    if testfn then
        local x = load_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size) --expand copies the input data to batch_size
        return x
    end
end

local function validdataset(batch_size)
    local x = load_data(validfn)
    x = replicate(x, batch_size)
    return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        vocab_map=vocab_map,
        invert_map = invert_map}
