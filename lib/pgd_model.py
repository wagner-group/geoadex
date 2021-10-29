import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection import proj_ball, proj_ball_intsct, proj_polytope_batch


def best_other_class(logits, exclude):
    """Returns the index of the largest logit, ignoring the class that
    is passed as `exclude`."""
    y_onehot = torch.zeros_like(logits)
    y_onehot.scatter_(1, exclude, 1)
    # make logits that we want to exclude a large negative number
    other_logits = logits - y_onehot * 1e9
    return other_logits.max(1)[0]


class PGDModel(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, params):
        super(PGDModel, self).__init__()
        self.basic_net = basic_net
        self.params = params

    def forward(self, inputs, targets, attack=False, params=None):
        if not attack:
            return self.basic_net(inputs)
        if not params:
            params = self.params

        # set network to eval mode to remove some training behavior (e.g.
        # drop out, batch norm)
        is_train = self.basic_net.training
        self.basic_net.eval()
        inputs_flat = inputs.view(-1, params['dim'])
        x = inputs.clone()
        losses = []

        if params['random_start']:
            x = x + torch.zeros_like(x).uniform_(-1, 1)
            # flat = (x - inputs).view(-1, params['dim'])
            # res = params['A'][:2000] @ flat[0] - params['b'][:2000]
            # print(res[res > 0])
            if params['constraint'] == 'balls':
                x = proj_ball_intsct(
                    x.view(-1, params['dim']), inputs.view(-1, params['dim']),
                    params['norms'], clip=params['clip'],
                    max_iters=params['max_proj_iter'], tol=1e-5).view(x.size())
            elif params['constraint'] == 'polytope':
                x = inputs + proj_polytope_batch(
                    (x - inputs).view(-1, params['dim']), params['A'],
                    params['AAT'], 1, params['b'], params,
                    clip_center=inputs_flat).view(x.size())

        for _ in range(params['num_steps']):
            # print(_)

            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)

                if params['loss_func'] == 'ce':
                    loss = F.cross_entropy(logits, targets, reduction='sum')
                elif params['loss_func'] == 'clipped_ce':
                    logsoftmax = torch.clamp(
                        F.log_softmax(logits, dim=1), np.log(params['gap']), 0)
                    loss = F.nll_loss(logsoftmax, targets, reduction='sum')
                elif params['loss_func'] == 'hinge':
                    other = best_other_class(logits, targets.unsqueeze(1))
                    loss = other - \
                        torch.gather(logits, 1, targets.unsqueeze(1)).squeeze()
                    # Positive gap creates stronger adv
                    loss = torch.min(
                        torch.tensor(params['gap']).cuda(), loss).sum()

            losses.append(loss.detach().cpu().item())

            # compute gradients
            grad = torch.autograd.grad(loss, x)[0].detach()
            # x = x.detach() + params['step_size'] * \
            #     F.normalize(grad.view(-1, params['dim']), 2, 1).view(x.size())
            x = x.detach() + params['step_size'] * grad
            # flat = (x - inputs).view(-1, params['dim'])
            # res = params['A'][:2000] @ flat[0] - params['b'][:2000]
            # print(res[res > 0])
            # projection step
            if params['constraint'] == 'balls':
                x = proj_ball_intsct(
                    x.view(-1, params['dim']), inputs.view(-1, params['dim']),
                    params['norms'], clip=params['clip'],
                    max_iters=params['max_proj_iter'], tol=1e-5).view(x.size())
            elif params['constraint'] == 'polytope':
                x = inputs + proj_polytope_batch(
                    (x - inputs).view(-1, params['dim']), params['A'],
                    params['AAT'], 1, params['b'], params,
                    clip_center=inputs_flat).view(x.size())
            # flat = (x - inputs).view(-1, params['dim'])
            # res = params['A'][:2000] @ flat[0] - params['b'][:2000]
            # print(res[res > 0])
            # print(x[x > 1].max())
            # print(x[x < 0].min())
            # import pdb
            # pdb.set_trace()

        self.basic_net.train(is_train)
        return self.basic_net(x), losses


class ADMMModel(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, params):
        super(ADMMModel, self).__init__()
        self.basic_net = basic_net
        self.params = params

    # def update_z(self, inputs, targets, v, params):
    #     delta = torch.zeros_like(inputs)
    #     delta.requires_grad_()
    #     optimizer = torch.optim.SGD(
    #         [delta], lr=params['lr_z'], momentum=0, weight_decay=0)
    #     for _ in range(params['max_z_iters']):
    #         optimizer.zero_grad()
    #         logits = self.basic_net(inputs + delta)
    #         # log_loss = torch.min(F.cross_entropy(
    #         #     logits, targets, reduction='mean'), torch.zeros(1, device='cuda') + 3)
    #         log_loss = F.cross_entropy(logits, targets, reduction='mean')
    #         aug_loss = ((delta - v) ** 2).sum()
    #         loss = - log_loss + len(params['norms']) * \
    #             params['rho'] / 2 * aug_loss
    #         loss.backward()
    #         optimizer.step()
    #         # import pdb
    #         # pdb.set_trace()
    #     del optimizer
    #     return delta.detach().view(-1, params['dim']), loss

    def update_z(self, inputs, targets, v, params, losses):
        # delta = torch.zeros_like(inputs)
        delta = v.clone()
        for _ in range(params['max_z_iters']):
            delta.requires_grad_()
            logits = self.basic_net(inputs + delta)
            log_loss = F.cross_entropy(logits, targets, reduction='sum')
            aug_loss = ((delta - v) ** 2).sum()
            # loss = - log_loss + len(params['norms']) * \
            #     params['rho'] / 2 * aug_loss
            loss = - log_loss + 2000 * params['rho'] / 2 * aug_loss
            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta = delta.detach() - params['lr_z'] * grad
            losses.append(log_loss.detach().cpu().item())
            # if grad.view(-1).norm() < params['tol']:
            #     break
            print(loss)

        return delta.detach().view(-1, params['dim']), loss

    def forward(self, inputs, targets, attack=False, params=None):
        if not attack:
            return self.basic_net(inputs)
        if not params:
            params = self.params

        # set network to eval mode to remove some training behavior (e.g.
        # drop out, batch norm)
        is_train = self.basic_net.training
        self.basic_net.eval()
        inputs_flat = inputs.view(-1, params['dim'])
        z = inputs.clone()
        losses = []

        if params['random_start']:
            z += torch.zeros_like(z).uniform_(-1, 1)
            if params['constraint'] == 'balls':
                z = proj_ball_intsct(
                    z.view(-1, params['dim']), inputs.view(-1, params['dim']),
                    params['norms'], clip=params['clip'],
                    max_iters=params['max_proj_iter'], tol=1e-5).view(z.size())
            elif params['constraint'] == 'polytope':
                # TODO
                z = inputs + proj_polytope_batch(
                    (z - inputs).view(-1, params['dim']), params['A'],
                    params['AAT'], 1, params['b'], params,
                    clip_center=inputs).view(z.size())

        if params['constraint'] == 'balls':
            z = (z - inputs).view(-1, params['dim'])
            if params['clip'] is not None:
                size = (len(params['norms']) + 1, z.size(0), params['dim'])
            else:
                size = (len(params['norms']), z.size(0), params['dim'])
        elif params['constraint'] == 'polytope':
            # NOTE: only implement to work with one sample (not batch)
            z = (z - inputs).view(params['dim'])
            if params['clip'] is not None:
                size = (params['A'].size(0) + 1, params['dim'])
            else:
                size = (params['A'].size(0), params['dim'])
        x = torch.zeros(size, device=params['device'], requires_grad=False)
        u = torch.zeros_like(x)

        for step in range(params['num_steps']):
            # update x_i
            if params['constraint'] == 'balls':
                for i, norm in enumerate(params['norms']):
                    x[i] = proj_ball(z - u[i], 0, norm[1], norm[0])
            elif params['constraint'] == 'polytope':
                # NOTE: only implement to work with one sample (not batch)
                if params['clip'] is not None:
                    diff = z - u[:-1]
                    res = (params['A'] * diff).sum(1) - params['b']
                    x[:-1] = diff - F.relu(res).unsqueeze(1) * params['A']
                else:
                    diff = z - u
                    res = (params['A'] * diff).sum(1) - params['b']
                    x = diff - F.relu(res).unsqueeze(1) * params['A']
            if params['clip'] is not None:
                x[-1] = torch.min(
                    torch.max(z - u[-1], params['clip'][0] - inputs_flat),
                    params['clip'][1] - inputs_flat)
            # update z
            v = (x.mean(0) + u.mean(0)).view(inputs.size())
            z, loss = self.update_z(inputs, targets, v, params, losses)
            # update u
            u += x - z
            print(step)
            aaa = params['A'] @ z.squeeze() - params['b']
            print(aaa.max())
            print(losses[-1])
        # print(loss)
        # import pdb
        # pdb.set_trace()
        # assert np.all(z.abs().sum(1).detach().cpu().numpy() <= 4)
        # assert np.all(z.norm(2, 1).detach().cpu().numpy() <= 3)
        # assert np.all(z.abs().max().detach().cpu().numpy() <= 0.1)
        # if np.any(z.abs().sum(1).detach().cpu().numpy() > 20):
        #     import pdb
        #     pdb.set_trace()
        # if np.any(z.norm(2, 1).detach().cpu().numpy() > 10):
        #     import pdb
        #     pdb.set_trace()
        # if np.any(z.abs().max().detach().cpu().numpy() > 0.5):
        #     import pdb
        #     pdb.set_trace()
        print(z.abs().sum(1))
        print(z.norm(2, 1))
        print(z.abs().max(1))
        import pdb
        pdb.set_trace()

        self.basic_net.train(is_train)
        return self.basic_net(inputs + z.view(inputs.size())), losses
