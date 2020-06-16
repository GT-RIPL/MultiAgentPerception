import torch

def get_vectorize_grad(model_parameter):
    grad_vec = None
    for i, para in enumerate(model_parameter):
        if para.requires_grad == True:
            grad = para.grad.view(para.grad.numel())
            if i == 0:
                grad_vec = grad
            else:
                grad_vec = torch.cat((grad_vec,grad), 0)
    return grad_vec
