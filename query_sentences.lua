-- query_sentences.lua

stringx = require('pl.stringx')
require 'io'
require('nngraph')
require('base')

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
                selection_mode = 'multinomial' -- 'top' or 'multinomial'
               }

function top_word(probs)
  -- return the most likely word 
  _,pred_id = torch.max(probs,1)
    return pred_id[1]
end

function multinom(probs)
  -- use multinomial distribution to select word. 
  -- more stochastic than top_word
    return torch.multinomial(probs, 1)[1]
end

function reset_state()
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function invert_map(map)
  -- invert map by switching the key, value pairs
    local inverse_map = {}
    for k,v in pairs(map) do
        inverse_map[v] = k
    end
    return inverse_map
end

function words2ids(word_sequence)
  -- uses vocab_map to convert word_sequence (table) to tensor of word indexes
  -- returns tensor of size input_length x batch_size

  id_sequence = {}
  for i,word in ipairs(word_sequence) do
    id_sequence[i]=vocab_map[word]
    if id_sequence[i]==nil then
      id_sequence[i] = vocab_map['<unk>'] --TODO: check if this is broken
    end
  end
  return id_sequence
end

function id2batch(w)
  -- duplicate word id into vector batch_size long
    return torch.Tensor(params.batch_size):fill(w)
end 

function write_sentence(word_table, output_length)
    -- prints sentence from the input tensor and the desired output length
    -- params:
      -- input table: table of word indexes, e.g. {1:5, 2:15, 3:46 4: 14}
      -- output length: number of words to add to sequence

    local word_ids = words2ids(word_table)
    local len = #word_ids + output_length --num data points in input sequence. 
    local word_id = nil
    

    -- initialize state, input x
    g_disable_dropout(model.rnns) --bc you're in test mode
    reset_state()
    g_replace_table(model.s[0], model.start_s)
    local x = torch.Tensor()
    local pred_id = nil
    local sentence = {}

    -- use each word to predict the next until len is reached
    for i = 1, len do 

      -- use word list and then use previous prediction as inputs
      if i<=#word_ids then
        word_id = word_ids[i]
        x = id2batch(word_id)
      else
        x = id2batch(pred_id)
      end
      
      --extract prediction and update state. 
      perp_tmp, model.s[1], log_probs = unpack(model.rnns[1]:forward({x, x, model.s[0]})) 
      g_replace_table(model.s[0], model.s[1])

      -- select top row and exponentiate to get actual probabilities
      probs = torch.exp(log_probs:select(1,1))

      -- convert probability to prediction
      if params.selection_mode == 'top' then
        pred_id = top_word(probs)
      elseif params.selection_mode == 'multinomial' then
        pred_id = multinom(probs)
      end
      sentence[i] = inverse_map[x[1]]
    end

    g_enable_dropout(model.rnns)
    return sentence
end

local function readline()
  -- reads user input and splits into output_length and table of words
  local line = io.read("*line")
  local output_length = nil
  local string = {}
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  output_length = table.remove(line,1)
  if tonumber(output_length) == nil then error({code="init"}) end
  for i = 1,#line do
    string[i] = line[i]:lower()
  end
  return {tonumber(output_length), string}
end


----------------
-- run
----------------
-- load and invert vocab map
vocab_map = torch.load(params.vocab_map_path)
inverse_map = invert_map(vocab_map)

-- load model
print 'loading model...'
model = torch.load(params.best_model_path)

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    output_length = line[1]
    sequence = line[2]
    print("Thanks, I will continue your sentence with " .. output_length .. " more words.")
    sentence = write_sentence(sequence, output_length)
    for i =1, #sentence do
      io.write(sentence[i]..' ')
    end
    io.write('\n')
    print(" ")
  end
end
