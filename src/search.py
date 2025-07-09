import logging

logging.basicConfig(level=logging.INFO)

import random
import numpy as np
import torch
from setting import config as cfg
from data.dataset import TrafficDataset
from model.mode import Mode
from model.nas import ALLOT
from model.architecture import Architecture
from run_manager import RunManager
from utils.helper import resort_candidate_op


def system_init(seed):
    """ Initialize random seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def main(num_epoch, load_mode=None, net_mode='all_path', model_name='ALLOT', opt='', seed='', rate=0.8, llm='LLAMA'):
    if len(seed) > 0:
        seed = int(seed)
        system_init(seed)
    else:
        system_init(cfg.sys.seed)

    # load data
    dataset = TrafficDataset(
        path=cfg.data.path,
        train_prop=cfg.data.train_prop,
        test_prop=cfg.data.test_prop,
        num_sensors=cfg.data.num_sensors,
        normalized_k=cfg.data.normalized_k,
        adj_type=cfg.data.adj_type,
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        batch_size=cfg.data.batch_size,
    )

    if 'LLM' in opt or 'Evo' in opt:
        weights = 1e-3*torch.randn(6, 3)
    else:
        weights = None

    net = ALLOT(
        adjinit=dataset.adj_mats[:,:,0],
        nodes=cfg.data.num_sensors,
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        in_size=cfg.data.in_size,
        out_size=cfg.data.out_size,
        hidden_size=cfg.model.hidden_size,
        skip_size=cfg.model.skip_size,
        layer_names=cfg.model.layer_names,
        skip_mode=cfg.model.skip_mode,
        node_out=cfg.model.node_out,
        num_nodes=cfg.model.num_nodes,
        candidate_op_profiles=cfg.model.candidate_op_profiles,
        dropout=cfg.model.dropout,
        opt=opt,
        weights=weights
    )

    print('# of weight parameters', len(list(net.weight_parameters())))
    print('# of arch parameters', len(list(net.arch_parameters())))
    print('# of proj parameters', len(list(net.proj_parameters())))
    print('# of parameters', len(list(net.parameters())))

    run_manager = RunManager(
        name=model_name,
        net=Architecture(net),
        dataset=dataset,

        arch_lr=cfg.trainer.arch_lr,
        arch_lr_decay_milestones=cfg.trainer.arch_lr_decay_milestones,
        arch_lr_decay_ratio=cfg.trainer.arch_lr_decay_ratio,
        arch_decay=cfg.trainer.arch_decay,
        arch_clip_gradient=cfg.trainer.arch_clip_gradient,

        weight_lr=cfg.trainer.weight_lr,
        weight_lr_decay_milestones=cfg.trainer.weight_lr_decay_milestones,
        weight_lr_decay_ratio=cfg.trainer.weight_lr_decay_ratio,
        weight_decay=cfg.trainer.weight_decay,
        weight_clip_gradient=cfg.trainer.weight_clip_gradient,

        num_search_epochs=cfg.trainer.num_search_epochs, 
        num_train_epochs=cfg.trainer.num_train_epochs,

        criterion=cfg.trainer.criterion,
        metric_names=cfg.trainer.metric_names,
        metric_indexes=cfg.trainer.metric_indexes,
        print_frequency=cfg.trainer.print_frequency,

        use_gpu=cfg.trainer.use_gpu,
        device_ids=cfg.trainer.device_ids,
        opt=opt,
        exp_mode='search',
        rate=rate,
        llm_model=llm
    )

    # run_manager._load(exp_mode='search')
    run_manager.initialize()
    net_modes = {
        'one_fixed':Mode.ONE_PATH_FIXED,
        'one_random':Mode.ONE_PATH_RANDOM,
        'two_path':Mode.TWO_PATHS,
        'all_path':Mode.ALL_PATHS,
        'project':Mode.PROJECT
    }
    net_mode =net_modes[net_mode]
    run_manager.search(num_epoch, net_mode=net_mode,
                       adjinit=dataset.adj_mats[:, :, 0],
                       nodes=cfg.data.num_sensors,
                       in_length=cfg.data.in_length,
                       out_length=cfg.data.out_length,
                       in_size=cfg.data.in_size,
                       out_size=cfg.data.out_size,
                       hidden_size=cfg.model.hidden_size,
                       skip_size=cfg.model.skip_size,
                       layer_names=cfg.model.layer_names,
                       skip_mode=cfg.model.skip_mode,
                       node_out=cfg.model.node_out,
                       num_nodes=cfg.model.num_nodes,
                       candidate_op_profiles=cfg.model.candidate_op_profiles,
                       dropout=cfg.model.dropout,
                       opt=opt,
                       weights=weights
                       )
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--load_mode', type=str, default='search') # search or final_project
    parser.add_argument('--net_mode', type=str, default='all_path')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--opt', type=str, default='')
    parser.add_argument('--seed', type=str, default='')
    parser.add_argument('--rate', type=float, default=0.8)
    parser.add_argument('--llm', type=str, default='LLAMA')

    args = parser.parse_args()

    cfg.load_config(args.config)
    model_name = cfg.model.name + '_' + args.desc + '_' + args.seed + '_' + str(args.rate)
    if len(args.opt) > 0:
        model_name += '_' + args.opt
    if len(args.llm) > 0 and args.llm != 'LLAMA':
        model_name += '_' + args.llm
    main(args.epoch, net_mode=args.net_mode, model_name=model_name, opt=args.opt, seed=args.seed, rate=args.rate, llm=args.llm)
