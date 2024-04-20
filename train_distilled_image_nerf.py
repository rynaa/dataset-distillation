import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from basics import task_loss, final_objective_loss, evaluate_steps
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results
from nerfpytorch.run_nerf import render_poses_distil, run_network, img2mse, load_blender_data
from nerf_dataset import ImagePoseDataset

def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]


class NerfTrainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        state.distill_steps = 1
        self.num_data_steps = state.distill_steps  # how much data we have # Number of images we are trying to learn
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        images, poses, render_poses, hwf, i_split = load_blender_data('./nerfpytorch/data/nerf_synthetic/lego', False, True, 8)
        
        # White BG
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        self.dataset = ImagePoseDataset(images, poses)
        state.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=16,
            num_workers=state.num_workers, pin_memory=True, shuffle=True
        )

        self.sampled_poses = poses[np.random.choice(poses.shape[0], 10)]
        
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        self.hwf = hwf
        print("HWF ", self.hwf)
        
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.K = K
         
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state

        # labels
        self.labels = []
        # distill_label = torch.arange(state.num_classes, dtype=torch.long, device=state.device) \
        #                      .repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...]]
        # distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        
        for _ in range(self.num_data_steps):
            self.labels.append(torch.from_numpy(self.sampled_poses).to(torch.device('cuda')))
        self.all_labels = torch.cat(self.labels)

        # data
        self.data = []
        for _ in range(self.num_data_steps):
            # TODO: Get blender data image size
            distill_data = torch.randn(self.sampled_poses.shape[0], 4, self.hwf[0], self.hwf[1],
                                       device=state.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # lr

        # undo the softplus + threshold
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        self.optimizer = optim.Adam(self.params, lr=state.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self):
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()

        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data, label, lr))

        return steps

    def forward(self, model, rdata, rlabel, steps):
        state = self.state

        # forward
        model.train()
        # model.to(torch.device('cuda'))
        w = model.get_param()
        params = [w]
        gws = []

        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=state.embed_fn,
                                                                embeddirs_fn=state.embeddirs_fn,
                                                                netchunk=64)
        render_kwargs = {
            'network_query_fn' : network_query_fn,
            'perturb' : False,
            'N_importance' : 0,
            'network_fine' : None,
            'N_samples' : 64,
            'network_fn' : lambda inputs: model.forward_with_param(inputs, w),
            'use_viewdirs' : True,
            'white_bkgd' : True,
            'raw_noise_std' : 0,
            'near': 2.0,
            'far': 6.0,
        }

        for step_i, (data, label, lr) in enumerate(steps):
            
            
            with torch.enable_grad():
                # Render
                rgbs, disps = render_poses_distil(label, self.hwf, self.K, 32, render_kwargs)
                #
                loss = img2mse(rgbs, data)
                print(loss)
            exit()
            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w
            render_kwargs['network_fn'] = lambda inputs: model.forward_with_param(inputs, w)

        # final L
        model.eval()
        render_kwargs['network_fn'] = lambda inputs: model.forward_with_param(inputs, params[-1])
        rrgbs, rdisps = render_poses_distil(rlabel, self.hwf, self.K, 32, render_kwargs)
        ll = final_objective_loss(rrgbs, rdata)
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []

        dw, = torch.autograd.grad(l, (params[-1],))

        # backward
        model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs

    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
        train_iter = iter(state.train_loader)
        for epoch in range(state.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < state.epochs - 1:
                    train_iter = iter(state.train_loader)
                yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device
        train_loader = state.train_loader
        sample_n_nets = state.local_sample_n_nets
        grad_divisor = state.sample_n_nets  # i.e., global sample_n_nets
        ckpt_int = state.checkpoint_interval

        data_t0 = time.time()

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0

            if it == 0:
                self.scheduler.step()

            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()
                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                # evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))

            do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)

            self.optimizer.zero_grad()
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            if sample_n_nets == state.local_n_nets:
                tmodels = self.models
            else:
                idxs = np.random.choice(state.local_n_nets, sample_n_nets, replace=False)
                tmodels = [self.models[i] for i in idxs]

            t0 = time.time()
            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:
                if state.train_nets_type == 'unknown_init':
                    model.reset(state)

                l, saved = self.forward(model, rdata, rlabel, steps)
                losses.append(l.detach())
                grad_infos.append(self.backward(model, rdata, rlabel, steps, saved))
                del l, saved
            self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            if state.distributed:
                all_reduce_coalesced(all_reduce_tensors, grad_divisor)
            else:
                for t in all_reduce_tensors:
                    t.div_(grad_divisor)

            # opt step
            self.optimizer.step()
            t = time.time() - t0

            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:  # nan
                    raise RuntimeError('loss became NaN')

            del steps, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps)
        return steps


def distill(state, models):
    return NerfTrainer(state, models).train()
