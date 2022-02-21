def train_step(hyp,  # path/to/hyp.yaml or hyp dictionary
               opt,
               device,
               callbacks):
    
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    
    
    return ...


def train_model(model, cfg):
    
    return ... 


