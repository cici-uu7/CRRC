import torch
from collections import OrderedDict

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device) for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        # 只对字典的值进行递归处理，跳过键
        return type(tensors)({name: tensors_to_device(tensor, device=device)
                             for name, tensor in tensors.items()})
    elif hasattr(tensors, '__getitem__') and hasattr(tensors, '__iter__'):
        # 处理类字典对象，如我们的 MetaBatch
        result = {}
        for key, value in tensors:
            result[key] = tensors_to_device(value, device=device)
        return result
    else:
        raise TypeError(f"不支持的数据类型: {type(tensors)}, 值: {tensors}")

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'