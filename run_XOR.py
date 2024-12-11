from tqdm import tqdm
from itertools import product
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from networks import *
from plot import *
from data import *


def evaluate_model(net, control_net, eval_loader, verbose_level=0):
    correct = 0
    total = 0
    with torch.no_grad():
        for eval_data, eval_labels, _ in eval_loader:
            net.reset_control_signals()
            if verbose_level >= 2:
                print("Eval", eval_data.shape, eval_labels)
            h1 = net.layer1(net.flatten(eval_data))
            output = net(eval_data)
            if verbose_level >= 2:
                print("Output", output)
            current_activities = torch.cat([net.flatten(eval_data), h1, output], dim=1)

            if verbose_level >= 2:
                print("Current activities", current_activities)

            control_signals = control_net(current_activities)
            net.set_control_signals(control_signals)
            if verbose_level >= 2:
                print("Control signals", net.hidden_activations.get_control_signals())
            outputs = net(eval_data)

            if verbose_level >= 2:
                print(outputs)

            predicted = outputs.max(1)[1]  # dim 1
            total += eval_labels.size(0)
            correct += (predicted == eval_labels.max(1)[1]).sum().item()

            if verbose_level >= 2:
                print(predicted)

    return 100 * correct / total


# -1: Print nothing
# 0: Print performance
# 1: Print loss per epoch
# 2: Print control signals, activities, outputs
verbose_level = 0
plot_data = False

# for task_id in range(1, 4):
#     print(f"Training on Task {task_id}")
#     dataloader = get_dataloader(task_id)
#     task_losses = []

#     if plot_data:
#         dataloader.dataset.plot()

#     for epoch in tqdm(range(num_epochs), desc=f"Task {task_id} Epochs", leave=False):
#         epoch_losses = []
#         for batch_data, batch_labels, _ in dataloader:
#             # Get current network activities
#             with torch.no_grad():
#                 net.reset_control_signals()
#                 h1 = net.layer1(net.flatten(batch_data))
#                 output = net(batch_data)
#                 current_activities = torch.cat(
#                     [net.flatten(batch_data), h1, output], dim=1
#                 )

#             # Inner loop - Training the control network
#             prev_loss = float("inf")
#             for inner_epoch in tqdm(
#                 range(inner_epochs), desc="Inner Epochs", leave=False
#             ):
#                 control_optimizer.zero_grad()

#                 control_signals = control_net(current_activities)
#                 if verbose_level >= 2:
#                     print("Control signals", control_signals.mean())
#                 net.set_control_signals(control_signals)

#                 output = net(batch_data)
#                 control_loss = criterion(output, batch_labels)
#                 l1_reg = l1_lambda * sum(
#                     (output - 1).abs().sum() for output in net(batch_data)
#                 )
#                 total_control_loss = control_loss + l1_reg

#                 total_control_loss.backward()
#                 control_optimizer.step()
#                 if abs(prev_loss - total_control_loss.item()) < control_threshold:
#                     if verbose_level >= 1:
#                         print("  Converged at epoch", inner_epoch)
#                     break

#                 prev_loss = total_control_loss.item()

#             # Update weights based on control signals
#             """
#             Above we trained the control network based on the current state of
#             the bio network. The control signal is at the beginning very strong
#             i.e. the control_net helps net a lot.
#             We now will update the weights of net. Then repeat this until the
#             network doesn't need any help anymore i.e. the control signal is
#             small.

#             We update weights using equiation 15 from the paper:

#                 delta_w = r_pre * phi( sum_i_pre[ w_i * r_i ] ) * ( a - a^* )

#             where:
#                 - delta_w: new weight change i.e. its gradient.
#                             E.g. net.layer1.weight.grad
#                 - r_pre:
#                 - phi: activation function
#                 - i_pre: presynaptic neuron index
#                 - w_i * r_i: postsynaptic potential (bias implied)
#                 - a: apical input
#                 - a^*: baseline apical input

#             additionally:
#                 - phi*a: r_post
#                 - phi*a^*: baseline r_post

#             Note: For the above we used the typical pyramidal neuron model:
#                 - apical:
#                     * receives control signal
#                     * Single dendrite receiving feedback signal from higher
#                         brain areas.
#                 - basal:
#                     * feedforward signals
#                     * Dendrites receiving feedforward signals from the previous
#                       layer.
#                 - axon:
#                     * r_pre
#                     * output of neuron
#                     * Long output dendrite transmitting signal to other neurons.

#             Note: TODO: Discuss bias

#             Note:
#                 - Naming convention pre and post is not based on the temporal flow
#                 - The terms presynaptic (pre) and postsynaptic (post) describe
#                   the relationship across a synapse:
#                 - Presynaptic neuron: The neuron sending the signal via its axon
#                   to another neuron.
#                 - Postsynaptic neuron: The neuron receiving the signal at its
#                   dendrites or soma.

#                 In short:
#                 - "input" => post
#                 - "output" => pre
#             """
#             if total_control_loss.item() > 0.01:
#                 with torch.no_grad():
#                     """
#                         batch_data.shape:
#                             * [batch_size, num_points, point_dim]
#                             * [32, 4, 2]

#                         net.flatten(batch_data).shape:
#                             * [batch_size, num_points * point_dim]
#                             * [32, 8]

#                         layer1.shape:
#                             * [batch_size, hidden_size]
#                             * [32, 20]

#                         layer2.shape:
#                             * [batch_size, output_size]
#                             * [32, 2]

#                         hidden_activations.shape:
#                             * [batch_size, hidden_size]
#                             * [32, 20]

#                         output_activations.shape:
#                             * [batch_size, output_size]
#                             * [32, 2]

#                         layer1.weight.grad.shape:
#                             * [hidden_size, input_size]
#                             * [20, 8]

#                         layer2.weight.grad.shape:
#                             * [output_size, hidden_size]
#                             * [2, 20]

#                     """

#                     """
#                     Implementation strategy:
#                         We basically have:

#                             dw = r_pre * r_post * (a - a^*)

#                         Now notice that ModulationReLULayer does

#                             self.control_signals * torch.relu(x)

#                         and notice that r_post is basically the output of the
#                         ModulatioNReLuLayer. Also remember that a is basically
#                         the control signal and a^* is the baseline. I assume
#                         the baseline is 1.0. We what we can do is create an
#                         "adjusted" control signal (a - a^*), apply it to our
#                         network and reduce the equation to

#                             dw = r_pre * r_post

#                         Basically we move (a - a^*) into r_post.

#                         Note: dw is the change in weight, not the actual update.
#                         So we set the gradient of the weight to dw and then use
#                         an optimizer to set the weight.

#                         Remember: An optimizer does e.g. gradient descent:

#                             weight = weight - lr * dw

#                         so we could do it by hand but why should be do that.
#                         (Sander told me to use an optimizer, so we use one ^^)
#                     """

#                     # Adjust control signal and set it
#                     a_diff = control_signals - torch.ones_like(control_signals)
#                     net.set_control_signals(a_diff)

#                     # Layer 1
#                     r_post_hidden = net.flatten(batch_data)
#                     r_pre_hidden = net.hidden_activations(net.layer1(r_post_hidden))

#                     dw = r_pre_hidden.T @ r_post_hidden
#                     net.layer1.weight.grad = dw

#                     # Layer 2
#                     r_post_output = r_pre_hidden
#                     r_pre_output = net.output_activations(net.layer2(r_post_output))

#                     dw = r_pre_output.T @ r_post_output
#                     net.layer2.weight.grad = dw

#                     # Update weights
#                     net_optimizer.step()

#                     # Get pre-synaptic activities
#                     # TODO
#                     # Calculate weight updates using the control-based rule
#                     # Layer 1 updates
#                     # TODO
#                     # Layer 2 updates
#                     # TODO

#                 epoch_losses.append(control_loss.item())

#         avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
#         task_losses.append(avg_epoch_loss)
#         if epoch % 1 == 0 and verbose_level >= 1:
#             print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

#     all_losses.extend(task_losses)

#     # Evaluation remains the same
#     task_performance[task_id] = {}
#     for eval_task_id in range(1, task_id + 1):
#         eval_loader = get_dataloader(eval_task_id)
#         accuracy = evaluate_model(net, control_net, eval_loader, verbose_level)
#         task_performance[task_id][eval_task_id] = accuracy
#         print(f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%")


def run_all_tasks(
    num_epochs,
    inner_epochs,
    learning_rate,
    control_lr,
    control_threshold,
    l1_lambda,
    verbose_level=-1,
    plot_data=False,
    seed=0,
):
    # Fix seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Net()
    control_net = ControlNet()
    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    all_losses = []
    task_performance = {}

    for task_id in tqdm(range(1, 4), desc="Tasks", disable=(verbose_level <= 0)):
        dataloader = get_dataloader(task_id)
        task_losses = []

        for epoch in tqdm(
            range(num_epochs),
            desc=f"Task {task_id} Epochs",
            leave=False,
            disable=(verbose_level <= 0),
        ):
            epoch_losses = []
            for batch_data, batch_labels, _ in dataloader:
                # Get current network activities
                with torch.no_grad():
                    net.reset_control_signals()
                    h1 = net.layer1(net.flatten(batch_data))
                    output = net(batch_data)
                    current_activities = torch.cat(
                        [net.flatten(batch_data), h1, output], dim=1
                    )

                # Inner loop - Training the control network
                prev_loss = float("inf")
                for inner_epoch in tqdm(
                    range(inner_epochs),
                    desc="Inner Epochs",
                    leave=False,
                    disable=(verbose_level <= 0),
                ):
                    control_optimizer.zero_grad()

                    control_signals = control_net(current_activities)
                    net.set_control_signals(control_signals)

                    if verbose_level >= 2:
                        print("Control signals", control_signals.mean())

                    output = net(batch_data)
                    control_loss = criterion(output, batch_labels)
                    l1_reg = l1_lambda * sum(
                        (output - 1).abs().sum() for output in net(batch_data)
                    )
                    total_control_loss = control_loss + l1_reg
                    total_control_loss.backward()
                    control_optimizer.step()
                    if abs(prev_loss - total_control_loss.item()) < control_threshold:
                        if verbose_level >= 1:
                            print("  Converged at epoch", inner_epoch)
                        break

                    prev_loss = total_control_loss.item()

                # Update weights based on control signals
                """
                Above we trained the control network based on the current state of
                the bio network. The control signal is at the beginning very strong
                i.e. the control_net helps net a lot.
                We now will update the weights of net. Then repeat this until the 
                network doesn't need any help anymore i.e. the control signal is
                small.

                We update weights using equiation 15 from the paper:

                    delta_w = r_pre * phi( sum_i_pre[ w_i * r_i ] ) * ( a - a^* )

                where:
                    - delta_w: new weight change i.e. its gradient.
                                E.g. net.layer1.weight.grad
                    - r_pre:
                    - phi: activation function
                    - i_pre: presynaptic neuron index
                    - w_i * r_i: postsynaptic potential (bias implied)
                    - a: apical input
                    - a^*: baseline apical input
                
                additionally:
                    - phi*a: r_post
                    - phi*a^*: baseline r_post

                Note: For the above we used the typical pyramidal neuron model:
                    - apical:
                        * receives control signal
                        * Single dendrite receiving feedback signal from higher
                            brain areas.
                    - basal:
                        * feedforward signals
                        * Dendrites receiving feedforward signals from the previous
                        layer.
                    - axon:
                        * r_pre
                        * output of neuron
                        * Long output dendrite transmitting signal to other neurons.
                
                Note: TODO: Discuss bias

                Note:
                    - Naming convention pre and post is not based on the temporal flow
                    - The terms presynaptic (pre) and postsynaptic (post) describe
                    the relationship across a synapse:
                    - Presynaptic neuron: The neuron sending the signal via its axon
                    to another neuron.
                    - Postsynaptic neuron: The neuron receiving the signal at its
                    dendrites or soma.
                    
                    In short:
                    - "input" => post
                    - "output" => pre
                """
                if total_control_loss.item() > 0.01:
                    with torch.no_grad():
                        """
                            batch_data.shape:
                                * [batch_size, num_points, point_dim]
                                * [32, 4, 2]
                            
                            net.flatten(batch_data).shape:
                                * [batch_size, num_points * point_dim]
                                * [32, 8]
                            
                            layer1.shape:
                                * [batch_size, hidden_size]
                                * [32, 20]
                            
                            layer2.shape:
                                * [batch_size, output_size]
                                * [32, 2]
                            
                            hidden_activations.shape:
                                * [batch_size, hidden_size]
                                * [32, 20]
                            
                            output_activations.shape:
                                * [batch_size, output_size]
                                * [32, 2]
                            
                            layer1.weight.grad.shape:
                                * [hidden_size, input_size]
                                * [20, 8]
                            
                            layer2.weight.grad.shape:
                                * [output_size, hidden_size]
                                * [2, 20]

                        """

                        """
                        Implementation strategy:
                            We basically have:

                                dw = r_pre * r_post * (a - a^*)
                            
                            Now notice that ModulationReLULayer does
                            
                                self.control_signals * torch.relu(x)
                            
                            and notice that r_post is basically the output of the
                            ModulatioNReLuLayer. Also remember that a is basically
                            the control signal and a^* is the baseline. I assume
                            the baseline is 1.0. We what we can do is create an
                            "adjusted" control signal (a - a^*), apply it to our
                            network and reduce the equation to

                                dw = r_pre * r_post

                            Basically we move (a - a^*) into r_post.

                            Note: dw is the change in weight, not the actual update.
                            So we set the gradient of the weight to dw and then use
                            an optimizer to set the weight.

                            Remember: An optimizer does e.g. gradient descent:
                            
                                weight = weight - lr * dw
                            
                            so we could do it by hand but why should be do that.
                            (Sander told me to use an optimizer, so we use one ^^)
                        """

                        # Adjust control signal and set it
                        a_diff = control_signals - torch.ones_like(control_signals)
                        net.set_control_signals(a_diff)

                        # Layer 1
                        r_post_hidden = net.flatten(batch_data)
                        r_pre_hidden = net.hidden_activations(net.layer1(r_post_hidden))

                        dw = r_pre_hidden.T @ r_post_hidden
                        net.layer1.weight.grad = dw

                        # Layer 2
                        r_post_output = r_pre_hidden
                        r_pre_output = net.output_activations(net.layer2(r_post_output))

                        dw = r_pre_output.T @ r_post_output
                        net.layer2.weight.grad = dw

                        # Update weights
                        net_optimizer.step()

                        # Get pre-synaptic activities
                        # TODO
                        # Calculate weight updates using the control-based rule
                        # Layer 1 updates
                        # TODO
                        # Layer 2 updates
                        # TODO

                    epoch_losses.append(control_loss.item())

            avg_epoch_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            )
            task_losses.append(avg_epoch_loss)
            if epoch % 1 == 0 and verbose_level >= 1:
                print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

        all_losses.extend(task_losses)

        task_performance[task_id] = {}
        for eval_task_id in range(1, task_id + 1):
            eval_loader = get_dataloader(eval_task_id)
            accuracy = evaluate_model(net, control_net, eval_loader, verbose_level)
            task_performance[task_id][eval_task_id] = accuracy
            if verbose_level >= 0:
                print(
                    f"Task {task_id} - Performance on Task {eval_task_id}: {accuracy:.2f}%"
                )

    return task_performance


def hyperparameter_search(
    num_epochs_list,
    inner_epochs_list,
    learning_rate_list,
    control_lr_list,
    control_threshold_list,
    l1_lambda_list,
):
    """
    Perform hyperparameter search over all combinations of given parameter lists.

    Args:
        num_epochs_list (list): List of possible values for num_epochs.
        inner_epochs_list (list): List of possible values for inner_epochs.
        learning_rate_list (list): List of possible values for learning_rate.
        control_lr_list (list): List of possible values for control_lr.
        control_threshold_list (list): List of possible values for control_threshold.
        l1_lambda_list (list): List of possible values for l1_lambda.

    Returns:
        best_params (dict): The best combination of hyperparameters.
        best_performance (dict): Performance for the best parameters.
    """
    total_combinations = (
        len(num_epochs_list)
        * len(inner_epochs_list)
        * len(learning_rate_list)
        * len(control_lr_list)
        * len(control_threshold_list)
        * len(l1_lambda_list)
    )
    # Combine all parameter lists into a single list of dictionaries
    all_combinations = product(
        num_epochs_list,
        inner_epochs_list,
        learning_rate_list,
        control_lr_list,
        control_threshold_list,
        l1_lambda_list,
    )

    best_params = None
    best_performance = None

    for combo in tqdm(
        all_combinations, desc="Hyperparameter Search", total=total_combinations
    ):
        # Unpack parameters
        (
            num_epochs,
            inner_epochs,
            learning_rate,
            control_lr,
            control_threshold,
            l1_lambda,
        ) = combo

        # Run tasks with the current hyperparameters
        performance = run_all_tasks(
            num_epochs=num_epochs,
            inner_epochs=inner_epochs,
            learning_rate=learning_rate,
            control_lr=control_lr,
            control_threshold=control_threshold,
            l1_lambda=l1_lambda,
        )

        # Check if all tasks achieved 100% accuracy
        if all(
            acc == 100 for task_acc in performance.values() for acc in task_acc.values()
        ):
            print("Found optimal parameters:", combo)
            return combo, performance

        # Update best params based on cumulative accuracy
        if best_performance is None or (
            sum(sum(perf.values()) for perf in performance.values())
            > sum(sum(perf.values()) for perf in best_performance.values())
        ):
            best_params = combo
            best_performance = performance

    return best_params, best_performance


def run_single_combination(combo):
    """
    Helper function to run a single combination of hyperparameters.
    """
    (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    ) = combo

    performance = run_all_tasks(
        num_epochs=num_epochs,
        inner_epochs=inner_epochs,
        learning_rate=learning_rate,
        control_lr=control_lr,
        control_threshold=control_threshold,
        l1_lambda=l1_lambda,
    )

    return combo, performance


def hyperparameter_search_parallel(
    num_epochs_list,
    inner_epochs_list,
    learning_rate_list,
    control_lr_list,
    control_threshold_list,
    l1_lambda_list,
):
    """
    Perform parallelized hyperparameter search over all combinations of given parameter lists.

    Args:
        num_epochs_list (list): List of possible values for num_epochs.
        inner_epochs_list (list): List of possible values for inner_epochs.
        learning_rate_list (list): List of possible values for learning_rate.
        control_lr_list (list): List of possible values for control_lr.
        control_threshold_list (list): List of possible values for control_threshold.
        l1_lambda_list (list): List of possible values for l1_lambda.

    Returns:
        best_params (dict): The best combination of hyperparameters.
        best_performance (dict): Performance for the best parameters.
    """
    total_combinations = (
        len(num_epochs_list)
        * len(inner_epochs_list)
        * len(learning_rate_list)
        * len(control_lr_list)
        * len(control_threshold_list)
        * len(l1_lambda_list)
    )
    all_combinations = list(
        product(
            num_epochs_list,
            inner_epochs_list,
            learning_rate_list,
            control_lr_list,
            control_threshold_list,
            l1_lambda_list,
        )
    )

    # Use multiprocessing pool to distribute work
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(run_single_combination, all_combinations),
                desc="Hyperparameter Search",
                total=total_combinations,
            )
        )

    # Find the best parameters
    best_params = None
    best_performance = None

    for combo, performance in results:
        if all(
            acc == 100 for task_acc in performance.values() for acc in task_acc.values()
        ):
            print("Found optimal parameters:", combo)
            return combo, performance

        if best_performance is None or (
            sum(sum(perf.values()) for perf in performance.values())
            > sum(sum(perf.values()) for perf in best_performance.values())
        ):
            best_params = combo
            best_performance = performance

    return best_params, best_performance


if __name__ == "__main__":
    """
    Best Parameters: (600, 50, 0.001, 0.001, 1e-09, 0.05)
    Best Performance: {1: {1: 100.0}, 2: {1: 60.0, 2: 100.0}, 3: {1: 100.0, 2: 96.0, 3: 100.0}}
    """
    # Define lists for each parameter
    num_epochs_list = [300, 600]
    inner_epochs_list = [50]
    learning_rate_list = [0.01, 0.001, 0.0001, 0.00001]
    control_lr_list = [0.01, 0.001, 0.0001, 0.00001]
    control_threshold_list = [1e-9]
    # Keep this small < 0.1 because otherwise the regularization term is too
    # strong and it fake converges
    l1_lambda_list = [0.001, 0.005, 0.01, 0.05, 0.1]

    # Perform hyperparameter search
    best_params, best_performance = hyperparameter_search_parallel(
        num_epochs_list,
        inner_epochs_list,
        learning_rate_list,
        control_lr_list,
        control_threshold_list,
        l1_lambda_list,
    )

    print("Best Parameters:", best_params)
    print("Best Performance:", best_performance)

    performance = run_all_tasks(*best_params, verbose_level=0)
    print("Run with best params:", performance)
