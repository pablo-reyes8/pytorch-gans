import torch 
import torch.nn.functional as F


def loss_hinge_discriminator(d_real, d_fake):
    """
    d_real: logits de D(x)   -> (N,1)
    d_fake: logits de D(G(z)) -> (N,1)
    """
    loss_real = torch.mean(F.relu(1.0 - d_real)) # max(0, 1 - D(x))
    loss_fake = torch.mean(F.relu(1.0 + d_fake)) # max(0, 1 + D(G(z)))
    return loss_real + loss_fake

def loss_hinge_generator(d_fake):
    """
    d_fake: logits de D(G(z))
    """
    return -torch.mean(d_fake) # G quiere subir D(G(z))


def r1_penalty(d_out_real, real):
    """
    R1 = E[ ||∂D/∂x||^2 ] sobre reales.
    """
    grad = torch.autograd.grad(
        outputs=d_out_real.sum(), inputs=real,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.reshape(grad.size(0), -1)
    return (grad.pow(2).sum(dim=1)).mean()