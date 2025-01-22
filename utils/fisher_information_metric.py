import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def compute_fisher_org(net, control_net, data_loader, with_signal=True, device="cpu"):
    """
    Compute the diagonal Fisher Information for 'net' w.r.t.
    cross-entropy loss on the given data_loader.
    Returns:
        A dict mapping param name -> FIM diagonal (tensor of same shape as param).
    """
    net.eval()
    net.to(device)

    fisher_dict = {}
    for name, param in net.named_parameters():
        fisher_dict[name] = torch.zeros_like(param, device=device)

    total_samples = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Reset control signals to ones to avoid dimension mismatch during forward pass
    net.reset_control_signals()

    for batch_data, batch_labels in data_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        net.zero_grad()
        control_net.zero_grad()

        if with_signal:
            net.reset_control_signals()
            with torch.no_grad():
                inp = net.flatten(batch_data)
                h1 = net.layer1(inp)
                foo = net.h1_mrelu(h1)
                h2 = net.layer2(foo)
                h3 = net.layer3(net.h2_mrelu(h2))
                output = net.layer4(net.h3_mrelu(h3))
                current_activities = torch.cat([inp, h1, h2, h3, output], dim=1)

                control_signals = control_net(current_activities)

            net.set_control_signals(control_signals)
        else:
            # Reset control signals again for each batch
            net.reset_control_signals()

        # net.reset_control_signals()
        outputs = net(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()

        bs = batch_data.size(0)
        total_samples += float(bs)

        for name, param in net.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data**2

    for name in fisher_dict:
        fisher_dict[name] /= total_samples

    net.train()
    return fisher_dict


def flatten_fisher_org(fisher_dict):
    """
    Flatten Fisher information from a dictionary into a single 1D numpy array,
    maintaining a consistent order of parameters.
    """
    # Sort keys to ensure consistent order across tasks
    sorted_items = sorted(fisher_dict.items(), key=lambda kv: kv[0])
    # Concatenate all flattened tensors and convert to numpy
    fisher_flat = torch.cat([param.flatten() for _, param in sorted_items])
    return fisher_flat.cpu().numpy()


def compute_fisher_net_and_control_net(net, control_net, dataloader):
    """
    Compute the diagonal Fisher Information for 'net' w.r.t.
    cross-entropy loss on the given data_loader.
    Returns:
        A dict mapping param name -> FIM diagonal (tensor of same shape as param).
    """
    # params1 = {
    #     "num_epochs": 30,
    #     "inner_epochs": 54,
    #     "learning_rate": 3.900671053825562e-06,
    #     "control_lr": 0.0008621989600943697,
    #     "control_threshold": 1.3565492056080836e-08,
    #     "l1_lambda": 0.0011869059296583477,
    # }

    # # Compute and print Fisher Information for Task 1
    # fisher_task1 = compute_fisher_org(net, dataloader)

    # # Flatten Fisher Information for plotting
    # fisher_flat1 = flatten_fisher(fisher_task1)

    # # Create a common x-axis based on the number of parameters from Task 1
    # param_indices = range(len(fisher_flat1))

    # # Set up two subplots, one for each task
    # fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    # # Plot Fisher Information for Task 1
    # axes.plot(param_indices, fisher_flat1, color="blue")
    # axes.set_title("Fisher Information for Task 1")
    # axes.set_ylabel("Fisher Information (F)")
    # axes.grid(True)
    # plt.tight_layout()
    # plt.show()
    net.eval()
    control_net.eval()

    fisher_dict_net = {}
    for name, param in net.named_parameters():
        fisher_dict_net[name] = torch.zeros_like(param)

    fisher_dict_control_net = {}
    for name, param in control_net.named_parameters():
        fisher_dict_control_net[name] = torch.zeros_like(param)

    total_samples = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Reset control signals to ones to avoid dimension mismatch during forward pass
    net.reset_control_signals()

    for batch_data, batch_labels, _ in dataloader:
        # batch_indices = batch_labels.argmax(dim=1)

        # Get current network activities
        with torch.no_grad():
            net.reset_control_signals()
            h1 = net.layer1(net.flatten(batch_data))
            output = net(batch_data)
            out = net.layer2(net.hidden_activations(h1))
            current_activities = torch.cat(
                [net.flatten(batch_data), h1, out, output], dim=1
            )

        control_signals = control_net(current_activities)
        net.set_control_signals(control_signals)
        output = net(batch_data)
        control_loss = criterion(output, batch_labels)
        # control_loss.backward()

        # TODO: Do we need reg?
        l1_reg = (
            0.0011869059296583477
            * (net(batch_data) - batch_labels).abs().sum(dim=1).mean()
        )

        total_control_loss = control_loss + l1_reg

        total_control_loss.backward()

        bs = batch_data.size(0)
        total_samples += float(bs)

        for name, param in net.named_parameters():
            if param.grad is not None:
                fisher_dict_net[name] += param.grad.data**2

        for name, param in control_net.named_parameters():
            if param.grad is not None:
                fisher_dict_control_net[name] += param.grad.data**2

    for name in fisher_dict_net:
        fisher_dict_net[name] /= total_samples

    for name in fisher_dict_control_net:
        fisher_dict_control_net[name] /= total_samples

    # model.train()

    return fisher_dict_net, fisher_dict_control_net


def flatten_fisher(fisher_dict):
    """
    Flatten Fisher information from a dictionary into a single 1D numpy array,
    maintaining a consistent order of parameters.
    """
    # Sort keys to ensure consistent order across tasks
    sorted_items = sorted(fisher_dict.items(), key=lambda kv: kv[0])
    # Concatenate all flattened tensors and convert to numpy
    fisher_flat = torch.cat([param.flatten() for _, param in sorted_items])

    return fisher_flat.cpu().numpy()


def plot_FIM(model, control_model, train_dataloaders):
    # compute_fisher_net_and_control_net(model, control_model, dataloader)
    fig, axes = plt.subplots(len(train_dataloaders), 1, figsize=(12, 10), sharex=True)

    for i, (dataloader, ax) in enumerate(zip(train_dataloaders, axes)):
        # return
        # fisher_net, fisher_control_net = compute_fisher_net_and_control_net(
        #     model, control_model, dataloader
        # )

        fisher_net = compute_fisher_org(model, control_model, dataloader)

        fisher_net_flat = flatten_fisher(fisher_net)

        # Flatten Fisher Information for plotting
        fisher_net_flat = flatten_fisher(fisher_net)

        # Create a common x-axis based on the number of parameters from Task 1
        param_indices = range(len(fisher_net_flat))

        # Set up two subplots, one for each task

        # Plot Fisher Information for Task 1
        ax.plot(param_indices, fisher_net_flat, color="blue")
        ax.set_title(f"Fisher Information of net for Task {i}")
        ax.set_ylabel("Fisher Information (F)")
        ax.grid(True)

    axes[-1].set_xlabel("Parameter Index (θ)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params1 = {
        "num_epochs": 30,
        "inner_epochs": 50,
        "learning_rate": 0.001,
        "control_lr": 0.001,
        "control_threshold": 1e-3,
        "l1_lambda": 0.01,
    }
    # Converting dict values to tuple as expected by run_all_tasks
    params_tuple = tuple(params1.values())
    params_used, perf, net = run_all_tasks(
        params_tuple, verbose_level=1, plot_data=True
    )
    avg(perf)

    # Compute and print Fisher Information for Task 1
    fisher_task1 = compute_fisher(net, get_dataloader(1))
    print("Fisher Information for Task 1:")
    for name, fisher_value in fisher_task1.items():
        print(f"{name}: {fisher_value.mean().item():.6f}")

    # Compute and print Fisher Information for Task 2
    fisher_task2 = compute_fisher(net, get_dataloader(2))
    print("Fisher Information for Task 2:")
    for name, fisher_value in fisher_task2.items():
        print(f"{name}: {fisher_value.mean().item():.6f}")

    # Flatten Fisher Information for plotting
    fisher_flat1 = flatten_fisher(fisher_task1)
    fisher_flat2 = flatten_fisher(fisher_task2)

    # Create a common x-axis based on the number of parameters from Task 1
    param_indices = range(len(fisher_flat1))

    # Set up two subplots, one for each task
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Fisher Information for Task 1
    aplt.plot(param_indices, fisher_flat1, color="blue")
    aplt.set_title("Fisher Information for Task 1")
    aplt.set_ylabel("Fisher Information (F)")
    aplt.grid(True)

    # Plot Fisher Information for Task 2
    plt.plot(param_indices, fisher_flat2, color="green")
    plt.set_title("Fisher Information for Task 2")
    plt.set_xlabel("Parameter Index (θ)")
    plt.set_ylabel("Fisher Information (F)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
