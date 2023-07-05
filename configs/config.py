from yacs.config import CfgNode as CN

def set_cfg(cfg):
    cfg.dataset = 'Boussinesq_convection_flow_3d'
    # data directory
    cfg.data_dir = './data/Boussinesq_convection_flow_3d'

    # model and equation
    cfg.model = 'spinn'
    cfg.equation = 'Boussinesq_convection_flow_3d'

    # input data settings
    cfg.nt = 32
    cfg.nxy = 128

    # training settings
    cfg.seed = 111
    cfg.lr = 1e-3
    cfg.epochs = 300000
    cfg.offset_num = 8
    cfg.offset_iter = 100
    cfg.lbda_c = 5000
    cfg.lbda_rho = 1000
    cfg.lbda_w = 1
    cfg.lbda_ic = 10000
    cfg.RBA = True

    # model settings
    cfg.mlp = 'modified_mlp'
    cfg.n_layers = 3
    cfg.features = 128
    cfg.r = 128
    cfg.out_dim = 3
    cfg.pos_enc = 5

    # time marching
    cfg.marching_steps = 10
    cfg.step_idx = 0
    cfg.time_end = 2.0

    # log settings
    cfg.log_iter = 1000
    cfg.plot_iter = 50000

    return cfg