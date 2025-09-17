import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule
import numpy as np


def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.

    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)
    
    # l = list(range(len(grads)))
    key_list = params.keys()
    params_list = list(params.values())
    
    params_new = OrderedDict(zip(key_list, params_list))
    # train_logit = model(train_input, params=params_new)
    # inner_loss = F.cross_entropy(train_logit, train_target)
    # model.zero_grad()
    # grads_new = torch.autograd.grad(inner_loss,
    #                                 params_new.values(),
    #                                 create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params_new.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params
    
def momentum_velocity(updated_params, momentum, velocity, beta1, beta2, meta_lr):
    
    init_keys = list(updated_params.keys())
    updated_params_list = list(updated_params.values())
    l = list(range(len(updated_params_list)))
    updated_params_mv = OrderedDict()
    
    for i in l:

        momentum[i] = beta1 * momentum[i] + (1 - beta1) * updated_params_list[i]
        velocity0 = velocity[i]
        velocity1 = beta2 * velocity0 + (1 - beta2) * torch.pow(updated_params_list[i], 2)
        velocity_hat = torch.from_numpy(np.maximum(velocity0.detach().numpy(), velocity1.detach().numpy()))
        # velocity_hat = torch.maximum(velocity0, velocity1)
        eta = torch.pow(torch.sqrt(velocity_hat), -1)
        velocity[i] = velocity1
        updated_params_mv[init_keys[i]] = updated_params_list[i] - meta_lr * momentum[i] * eta
        
    return updated_params_mv
    
    
    
    
    
    