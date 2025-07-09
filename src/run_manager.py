import logging
import os

import torch
import torch.utils
import numpy as np

import logging

import utils.metric
from model.mode import Mode
from setting import param_path
from model.smooth_dart_optimizer import Linf_PGD_alpha, Random_alpha
from model.LLM import LLM
from model.Evolution import Evolution
from model.nas import ALLOT

class RunManager:
    def __init__(self,
                 name,
                 net,
                 dataset,
                 arch_lr, arch_lr_decay_milestones, arch_lr_decay_ratio, arch_decay, arch_clip_gradient,
                 weight_lr, weight_lr_decay_milestones, weight_lr_decay_ratio, weight_decay, weight_clip_gradient,
                 num_search_epochs, num_train_epochs,
                 criterion, metric_names, metric_indexes,
                 print_frequency,
                 use_gpu, device_ids, opt='', dataset_name='', llm_model='LLAMA', llm_layers=3, exp_mode='train',
                 num_block=1, num_nodes=4, num_opt=3, rate=0.8, ):

        self._name = name
        self._dataset = dataset
        self._net = self.net_mount_device(net, use_gpu, device_ids)

        # arch optimizer
        self._arch_lr = arch_lr
        self._arch_lr_decay_milestones = arch_lr_decay_milestones
        self._arch_lr_decay_ratio = arch_lr_decay_ratio
        self._arch_decay = arch_decay
        self._arch_clip_gradient = arch_clip_gradient

        # nn optimizer
        self._weight_lr = weight_lr
        self._weight_lr_decay_milestones = weight_lr_decay_milestones
        self._weight_lr_decay_ratio = weight_lr_decay_ratio
        self._weight_decay = weight_decay
        self._weight_clip_gradient = weight_clip_gradient

        self._num_search_epochs = num_search_epochs
        self._num_train_epochs = num_train_epochs

        self._criterion = getattr(utils.metric, criterion)
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency

        self._cross_tag = False

        self.opt = opt
        self.perturb_alpha = None

        if 'LLM' in opt and exp_mode == 'search':
            if 'DFS' in opt:
                option = 'DFS'
            else:
                option = 'straight'
            self.LLM = LLM(dataset_name=dataset_name, device=self._device, llm_model=llm_model, llm_layers=llm_layers, option=option, rate=rate).to(self._device)

        if 'Evo' in opt and exp_mode == 'search':
            if 'Age' in opt:
                strategy = 'Age'
            else:
                strategy = 'Normal'
            self.Evolution = Evolution(strategy=strategy, device=self._device).to(self._device)

    def update_name(self, name):
        self._name = name

    def set_cross_tag(self, tag):
        self._cross_tag = tag
        
    def net_mount_device(self, net, use_gpu, device_ids):
        use_gpu = (True if torch.cuda.is_available() else False) and use_gpu
        if use_gpu:
            #if os.environ.get('CUDA_VISIBLE_DEVICES'):
            #    device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
            logging.info('Use GPU: cuda:{}'.format(device_ids[0]))
            use_devices = ','.join([str(d_id) for d_id in device_ids])
            os.environ["CUDA_VISIBLE_DEVICES"] = use_devices
            device = torch.device('cuda:{}'.format(device_ids[0]))
            if len(device_ids)>1:
                net = torch.nn.DataParallel(net, device_ids=device_ids)
        else:
            logging.info('Use CPU')
            device = torch.device('cpu')
        self._device = device
        self._device_ids = device_ids
        logging.info(self._device)
        return net.to(device)    

    def _save(self, exp_mode):
        save_dir = os.path.join(param_path, self._name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        states = {
            'net': self._net.state_dict(),
            'arch_optimizer': self._arch_optimizer.state_dict(),
            'arch_optimizer_scheduler': self._arch_optimizer_scheduler.state_dict(),
            'weight_optimizer': self._weight_optimizer.state_dict(),
            'weight_optimizer_scheduler': self._weight_optimizer_scheduler.state_dict(),
            'best_epoch': self._best_epoch,
            'valid_records': self._valid_records
        }
        filename = os.path.join(save_dir, '{}.pth'.format(exp_mode))
        torch.save(obj = states, f = filename)
        logging.info('[eval]\t epoch[{}]\t save parameters to {}'.format(self._best_epoch, filename))
        logging.info(self._net)

    def _save_checkpoint(self, epoch, exp_mode):
        save_dir = os.path.join(param_path, self._name, 'checkpoint-{}'.format(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        states = {
            'net': self._net.state_dict(),
            'arch_optimizer': self._arch_optimizer.state_dict(),
            'arch_optimizer_scheduler': self._arch_optimizer_scheduler.state_dict(),
            'weight_optimizer': self._weight_optimizer.state_dict(),
            'weight_optimizer_scheduler': self._weight_optimizer_scheduler.state_dict(),
            'best_epoch': self._best_epoch,
            'valid_records': self._valid_records
        }
        filename = os.path.join(save_dir, '{}.pth'.format(exp_mode))
        torch.save(obj = states, f = filename)
        logging.info('save checkpoint-{} to {}'.format(epoch, filename))

    def _load(self, exp_mode):
        # initialize for optimizers and clear validation records
        self.initialize()
        save_dir = os.path.join(param_path, self._name)
        filename = os.path.join(save_dir, '{}.pth'.format(exp_mode))
        logging.info(filename)
        try:
            states = torch.load(filename)
            # load net
            if self._cross_tag:
                pred_dic = {k: v for k, v in states['net'].items() if
                            ("nodevec" not in k and "spatil_embedding" not in k)}
            else:
                pred_dic = states['net']
            self._net.load_state_dict(pred_dic, strict=False)
            # self._net.load_state_dict(states['net'])
            logging.info(self._net)
            # load optimizer
            self._arch_optimizer.load_state_dict(states['arch_optimizer'])
            self._arch_optimizer_scheduler.load_state_dict(states['arch_optimizer_scheduler'])
            self._weight_optimizer.load_state_dict(states['weight_optimizer'])
            self._weight_optimizer_scheduler.load_state_dict(states['weight_optimizer_scheduler'])
            # load historical records
            self._best_epoch = states['best_epoch']
            self._valid_records = states['valid_records']
            logging.info('load architecture [epoch {}] from {} [ok]'.format(self._best_epoch, filename))
        except:
            logging.info('load architecture [fail]')
            logging.info('initialize the optimizer')
            self.initialize()

    def clear_records(self):
        self._best_epoch = -1
        self._valid_records = []

    def initialize(self):
        # initialize for weight optimizer
        if 'linf' in self.opt:
            self.perturb_alpha = Linf_PGD_alpha
        elif 'rand' in self.opt:
            self.perturb_alpha = Random_alpha

        self._weight_optimizer = torch.optim.Adam(
            self._net.weight_parameters(),
            lr=self._weight_lr,
            weight_decay=self._weight_decay
        )
        self._weight_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._weight_optimizer,
            milestones=self._weight_lr_decay_milestones,
            gamma=self._weight_lr_decay_ratio,
        )
        # initialize for arch optimizer
        self._arch_optimizer = torch.optim.Adam(
            self._net.arch_parameters(),
            lr=self._arch_lr,
            weight_decay=self._arch_decay
        )
        self._arch_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._arch_optimizer,
            milestones=self._arch_lr_decay_milestones,
            gamma=self._arch_lr_decay_ratio,
        )
        # initialize validation records
        self.clear_records()

    def initialize_param_only(self):
        # initialize for weight optimizer
        if 'linf' in self.opt:
            self.perturb_alpha = Linf_PGD_alpha
        elif 'rand' in self.opt:
            self.perturb_alpha = Random_alpha

        self._weight_optimizer = torch.optim.Adam(
            self._net.weight_parameters(),
            lr=self._weight_lr,
            weight_decay=self._weight_decay
        )
        self._weight_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._weight_optimizer,
            milestones=self._weight_lr_decay_milestones,
            gamma=self._weight_lr_decay_ratio,
        )
        # initialize for arch optimizer
        self._arch_optimizer = torch.optim.Adam(
            self._net.arch_parameters(),
            lr=self._arch_lr,
            weight_decay=self._arch_decay
        )
        self._arch_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._arch_optimizer,
            milestones=self._arch_lr_decay_milestones,
            gamma=self._arch_lr_decay_ratio,
        )


    def _add_record(self, metrics, exp_mode):
        self._valid_records += [metrics.get_value()]
        best_valid_records = self._valid_records[self._best_epoch][self._metric_names[0]]
        last_valid_records = self._valid_records[-1][self._metric_names[0]]
        if self._best_epoch < 0 or best_valid_records > last_valid_records:
            self._best_epoch = len(self._valid_records) - 1
            self._save(exp_mode)


    def _save_record(self, metrics, exp_mode):
        results = metrics.get_value()
        
        save_dir = os.path.join(param_path, self._name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        filename = os.path.join(save_dir, 'metrics_{}.pkl'.format(exp_mode))

        import pickle as pkl
        pkl.dump(results, open(filename, 'wb'))

    def _train_epoch(self, epoch, train_loader, tag='train', net_mode=Mode.ALL_PATHS, weights=None):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset._batch_size
        )

        self._net.train()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
            self._weight_optimizer.zero_grad()

            batch_x = batch_x.float().to(self._device)
            batch_x_mark = batch_x_mark.float().to(self._device)

            preds = self._net(batch_x, batch_x_mark, 
                        attn_mask = None, adj_mats = self._dataset.adj_mats, 
                        mode = net_mode, weights = weights)
            preds = self._dataset.scaler.inverse_transform(preds)
            # batch_y = self._dataset.scaler.inverse_transform(batch_y)
            loss = self._criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.weight_parameters(), self._weight_clip_gradient)
            self._weight_optimizer.step()
            # log metrics
            speedometer.update(preds, batch_y)

        logging.info('-'*30)
        logging.info('Train epoch [{}] finished'.format(epoch))
        logging.info(self._net)
        logging.info('-'*30)

        return speedometer.finish()

    def train(self, num_train_epochs=None, net_mode=Mode.ONE_PATH_FIXED):
        self.clear_records()

        train_loader = self._dataset.get_dataloader(tag='train')
        valid_loader = self._dataset.get_dataloader(tag='valid')
        test_loader  = self._dataset.get_dataloader(tag='test')

        num_train_epochs = num_train_epochs or self._num_train_epochs

        for epoch in range(num_train_epochs):
            self._train_epoch(epoch, train_loader, net_mode=net_mode)
            valid_metrics = self.evaluate(epoch, valid_loader, tag='valid', net_mode=net_mode)
            self._add_record(valid_metrics, exp_mode='train')

            self._weight_optimizer_scheduler.step()

        self._load('train')

        test_metrics = self.evaluate(epoch, test_loader, tag='test', net_mode=net_mode)
        self._save_record(test_metrics, 'train')


    def test(self, net_mode=Mode.ONE_PATH_FIXED):
        self.clear_records()
        test_loader  = self._dataset.get_dataloader(tag='test')
        test_metrics = self.evaluate(0, test_loader, tag='test', net_mode=net_mode)
        
        self._save_record(test_metrics, 'test')


    def evaluate(self, epoch, dataloader, tag, net_mode=Mode.ALL_PATHS, weights=None):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset._batch_size
        )
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(dataloader):
            with torch.no_grad():
                self._net.eval()
                
                batch_x = batch_x.float().to(self._device)
                batch_x_mark = batch_x_mark.float().to(self._device)
                
                preds = self._net(batch_x, batch_x_mark, 
                            attn_mask = None, adj_mats = self._dataset.adj_mats, 
                            mode = net_mode, weights = weights)
                preds = self._dataset.scaler.inverse_transform(preds)
                # batch_y = self._dataset.scaler.inverse_transform(batch_y)
            # log metrics
            speedometer.update(preds, batch_y)
            
        # self._net.train()

        logging.info('-'*30)
        logging.info('Epoch [{}] and Tag [{}] finished'.format(epoch, tag))
        logging.info(self._net)
        logging.info('-'*30)
        
        return speedometer.finish()

    def _search_epoch(self, epoch, search_train_loader, search_valid_loader, tag='search', net_mode=Mode.ALL_PATHS, weights=None):
        speedometer = Speedometer(
            title=tag,
            epoch=epoch,
            metric_names=self._metric_names,
            metric_indexes=self._metric_indexes,
            print_frequency=self._print_frequency,
            batch_size=self._dataset._batch_size
        )
        weight_lr = self._weight_optimizer_scheduler.get_lr()[0]
        arch_lr = self._arch_optimizer_scheduler.get_lr()[0]
        logging.info('Search epoch [{}], weight lr:{}, arch lr: {}'.format(epoch, weight_lr, arch_lr))

        # self._net.train()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(search_train_loader):
            ###### train arch weights
            self._net.eval()
            self._weight_optimizer.zero_grad(); self._arch_optimizer.zero_grad()

            batch_x_arch,batch_y_arch,batch_x_mark_arch,batch_y_mark_arch = next(iter(search_valid_loader))
            batch_x_arch = batch_x_arch.float().to(self._device)
            batch_x_mark_arch = batch_x_mark_arch.float().to(self._device)

            preds = self._net(batch_x_arch, batch_x_mark_arch, 
                        attn_mask = None, adj_mats = self._dataset.adj_mats, 
                        mode = net_mode, weights = weights)
            preds = self._dataset.scaler.inverse_transform(preds)
            # batch_y = self._dataset.scaler.inverse_transform(batch_y)
            loss = self._criterion(preds, batch_y_arch)
            loss.backward() # (retain_graph=False)

            torch.nn.utils.clip_grad_norm_(self._net.arch_parameters(), self._arch_clip_gradient)
            self._arch_optimizer.step()

            if self.perturb_alpha:
                self.perturb_alpha(self._net, batch_x_arch, batch_x_mark_arch, self.epsilon_alpha)

            ##################################################
            ###### train param weights
            self._net.train()
            self._weight_optimizer.zero_grad(); self._arch_optimizer.zero_grad()

            batch_x = batch_x.float().to(self._device)
            batch_x_mark = batch_x_mark.float().to(self._device)

            preds = self._net(batch_x, batch_x_mark, 
                        attn_mask = None, adj_mats = self._dataset.adj_mats, 
                        mode = net_mode, weights = weights)
            preds = self._dataset.scaler.inverse_transform(preds)
            # batch_y = self._dataset.scaler.inverse_transform(batch_y)
            loss = self._criterion(preds, batch_y)
            loss.backward() # (retain_graph=False)
            
            torch.nn.utils.clip_grad_norm_(self._net.weight_parameters(), self._weight_clip_gradient)
            self._weight_optimizer.step()
            
            # log metrics
            speedometer.update(preds, batch_y)


        logging.info('-'*30)
        logging.info('Search epoch [{}] finished'.format(epoch))
        logging.info(self._net)
        logging.info('-'*30)

        return speedometer.finish()

    def search(self, num_search_epochs=None, net_mode=Mode.ALL_PATHS,
               adjinit=None,
               nodes=None,
               in_length=None,
               out_length=None,
               in_size=None,
               out_size=None,
               hidden_size=None,
               skip_size=None,
               layer_names=None,
               skip_mode=None,
               node_out=None,
               num_nodes=None,
               candidate_op_profiles=None,
               dropout=None,
               opt=None,
               weights=None
               ):
        self.clear_records()

        train_loader = self._dataset.get_dataloader(tag='train')
        valid_loader = self._dataset.get_dataloader(tag='valid')
        test_loader  = self._dataset.get_dataloader(tag='test')

        num_search_epochs = num_search_epochs or self._num_search_epochs

        for epoch in range(self._best_epoch + 1, num_search_epochs):
            if self.perturb_alpha is not None:
                self.epsilon_alpha = 0.03 + (0.3 - 0.03) * epoch / num_search_epochs
            if 'LLM' in self.opt or 'Evo' in self.opt:
                self._train_epoch(epoch, train_loader, net_mode=Mode.ONE_PATH_FIXED)
            else:
                self._search_epoch(epoch, train_loader, valid_loader, net_mode=net_mode)
            valid_metrics = self.evaluate(epoch, valid_loader, tag='valid', net_mode=Mode.ONE_PATH_FIXED)
            self._add_record(valid_metrics, exp_mode='search')

            self._weight_optimizer_scheduler.step()
            self._arch_optimizer_scheduler.step()

            
            if epoch % 10 ==0:
                self._save_checkpoint(epoch, exp_mode='search')

            if 'LLM' in self.opt or 'Evo' in self.opt:
                LLM_start = time.time()
                if 'LLM' in self.opt:
                    new_weights = self.LLM(weights, valid_metrics, epoch, num_search_epochs)
                else:
                    new_weights = self.Evolution(weights, valid_metrics)
                if new_weights is not None:
                    weights = new_weights
                self._net.reset_model(ALLOT(
                    adjinit=adjinit,
                    nodes=nodes,
                    in_length=in_length,
                    out_length=out_length,
                    in_size=in_size,
                    out_size=out_size,
                    hidden_size=hidden_size,
                    skip_size=skip_size,
                    layer_names=layer_names,
                    skip_mode=skip_mode,
                    node_out=node_out,
                    num_nodes=num_nodes,
                    candidate_op_profiles=candidate_op_profiles,
                    dropout=dropout,
                    opt=opt,
                    weights=weights
                ).to(self._device))
                self.initialize_param_only()
                logging.info("LLM Time:{}".format(time.time() - LLM_start))
                

        self._load('search')

        test_metrics = self.evaluate(epoch, test_loader, tag='test', net_mode=Mode.ONE_PATH_FIXED)
        self._save_record(test_metrics, 'search')

    def project(self, epoch, cell_id, proj_loader, num_cells=3, edge_decision='random', project_criterion='loss'):
        num_op = self._net.num_ops(cell_id)
        
        remain_eids = self._net.remain_edge_ids(cell_id)
        if edge_decision == 'random':
            selected_eid = np.random.choice(remain_eids, size=1)[0]
        else:
            selected_eid = np.random.choice(remain_eids, size=1)[0]

        if project_criterion == 'loss':
            compare = lambda x,y: x > y
        elif project_criterion == 'acc':
            compare = lambda x,y: x < y
        else:
            compare = lambda x,y: x > y

        best_opid = 0
        crit_extrema = None
        for op_id in range(num_op):
            logging.info('Project epoch: [{}], cell id: {}, run op: {}'.format(epoch, cell_id, op_id))
            
            weights_total = self._net.project_masked_weights(cell_id, selected_eid, op_id)
            project_tag = 'proj cell_id:{} e_id:{} op_id:{}'.format(cell_id, selected_eid, op_id)
            
            crit = self.evaluate(epoch, proj_loader, net_mode=Mode.ALL_PATHS, tag=project_tag, weights=weights_total)
            crit = crit.get_value()['mae'][-1]
            if crit_extrema is None or compare(crit, crit_extrema):
                crit_extrema = crit
                best_opid = op_id
        
        return selected_eid, best_opid

    def pt_project(self, project_init):
        self.clear_records()

        num_cells = self._net.num_cells()
        num_layers= self._net.num_layers()
        num_edges = self._net.num_edges()
        logging.info('Num stcells:{} total, Num edges: {} in each cell'.format(num_cells, num_edges))

        train_loader = self._dataset.get_dataloader(tag='train')
        valid_loader = self._dataset.get_dataloader(tag='valid')
        test_loader  = self._dataset.get_dataloader(tag='test')

        self._net.train()

        # proj_intv
        tune_epochs = project_init * (num_edges) * (num_cells)
        logging.info('Project tune epochs: {}'.format(tune_epochs))

        cell_id = 0
        for epoch in range(tune_epochs):
            if epoch % project_init==0 or epoch==(tune_epochs-1):
                cell_id = self._net.next_cell(cell_id)
                
                logging.info('Project epoch: [{}], next cell_id:'.format(epoch, cell_id))
            
                if cell_id >= num_layers:
                    logging.info('Cell id is out of index [{}]'.format(num_layers))
                    continue

                logging.info('Project epoch: [{}], cell_id (stcell or stlayer): {}'.format(epoch, cell_id))
                
                selected_eid, best_opid = self.project(epoch, cell_id, valid_loader, num_cells=num_cells)
                
                logging.info('Project epoch: [{}], selected_eid: {} best_opid: {}'.format(epoch, selected_eid, best_opid))
                
                self._net.project_op(cell_id, selected_eid, best_opid)

                logging.info(self._net)
            
            proj_tag = 'Project cell_id:{} e_id:{} op_id:{}'.format(cell_id, selected_eid, best_opid)
            self._search_epoch(epoch, train_loader, valid_loader, tag=proj_tag, net_mode=Mode.PROJECT, weights=None)

            # train_metrics = self.evaluate(epoch, train_loader, tag='train project', net_mode=Mode.PROJECT)
            # valid_metrics = self.evaluate(epoch, valid_loader, tag='valid project', net_mode=Mode.PROJECT)
            test_metrics = self.evaluate(epoch, test_loader, tag='test project', net_mode=Mode.PROJECT)

            self._add_record(test_metrics, exp_mode='project')

        self._save(exp_mode='final_project')

        logging.info('-'*30)
        logging.info('Project total epochs: [{}] finished'.format(tune_epochs))
        logging.info(self._net)
        logging.info('-'*30)

        self._load('final_project')
        test_metrics = self.evaluate(epoch, test_loader,  tag='test', net_mode=Mode.PROJECT)
        self._save_record(test_metrics, 'final_project')


import time

class Speedometer:
    def __init__(self, title, epoch, metric_names, metric_indexes, print_frequency, batch_size):
        self._title = title
        self._epoch = epoch
        self._metric_names = metric_names
        self._metric_indexes = metric_indexes
        self._print_frequency = print_frequency
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._metrics = utils.metric.Metrics(self._metric_names, self._metric_indexes)
        self._start = time.time()
        self._tic = time.time()
        self._counter = 0

    def update(self, preds, labels, step_size=1):
        self._metrics.update(preds, labels)
        self._counter += step_size
        if self._counter % self._print_frequency == 0:
            time_spent = time.time() - self._tic
            speed = float(self._print_frequency * self._batch_size) / time_spent
            out_str = [
                '[{}]'.format(self._title),
                'epoch[{}]'.format(self._epoch),
                'batch[{}]'.format(self._counter),
                'time: {:.2f}'.format(time_spent),
                'speed: {:.2f} samples/s'.format(speed),
                str(self._metrics)
            ]
            logging.info('\t'.join(out_str))
            self._tic = time.time()

    def finish(self):
        out_str = [
            '[{}]'.format(self._title),
            'epoch[{}]'.format(self._epoch),
            'time: {:.2f}'.format((time.time() - self._start)),
            str(self._metrics)
        ]
        logging.info('\t'.join(out_str))
        return self._metrics
