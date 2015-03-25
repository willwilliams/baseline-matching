local M = { }

function M.parse(arg)
    local defaultDir = '/neural_networks'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache',
               defaultDir ..'/parallel_stack_baseline_runs',
               'subdirectory in which to save/log experiments')
    cmd:option('-trainData',
               '~/',
               'home of training streetview dataset')
    cmd:option('-testData', 
               defaultDir .. '/', 
               'home of testing streetview dataset')
    cmd:option('-machines',
               'machines.txt',  
               'list of machine IPs to train on')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       10000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size for each machine')
    cmd:option('-nTest',           1000,  'number of test images to use')
    cmd:option('-testBatchSize',    12,   'mini-batch size for testing')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'alexnet', 'Options: alexnet | overfeat')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string('parallelstack', opt,
                                       {retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, ',' .. os.date():gsub(' ',''))
    return opt
end

return M
