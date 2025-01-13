import torch


# TODO: Save optimizer state as well and maybe meta info like seed etc.
def save_model_with_grads(model, save_path):
    """
    Save a model with gradients for each parameter.
    """

    # TODO: Maybe we need to do a forward pass in eval mode and disable
    # control signals?
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise ValueError("Model parameters must have gradients to save.")
        grads[f"{name}.grad"] = param.grad.data.cpu()

    res = {"model": model.state_dict(), "grads": grads}

    torch.save(res, save_path)
