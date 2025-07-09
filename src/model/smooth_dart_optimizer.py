import torch
from torch.optim.optimizer import Optimizer


class Linf_SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Linf_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Linf_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                #d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss



# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD_alpha(model, X, y, epsilon, steps=7, random_start=True):
    training = model.training
    if training:
        model.eval()
    saved_params = [p.clone() for p in model.arch_parameters()]
    optimizer = Linf_SGD(model.arch_parameters(), lr=2 * epsilon / steps)
    with torch.no_grad():
        loss_before = model._loss(X, y, updateType='weight')
    if random_start:
        for p in model.arch_parameters():
            p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
        model.clip()

    for _ in range(steps):
        optimizer.zero_grad()
        model.zero_grad()
        loss = -model._loss(X, y, updateType='weight')
        loss.backward()
        optimizer.step()
        diff = [(model.arch_parameters()[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))]
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(diff[i] + saved_params[i])
        model.clip()

    optimizer.zero_grad()
    model.zero_grad()
    with torch.no_grad():
        loss_after = model._loss(X, y, updateType='weight')
    if loss_before > loss_after:
        for i, p in enumerate(model.arch_parameters()):
            p.data.copy_(saved_params[i])
    if training:
        model.train()


def Random_alpha(model, X, y, epsilon):
    for p in model.arch_parameters():
        p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
    model.clip()