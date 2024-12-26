import itertools
from tqdm import tqdm
import time
from itertools import product
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, Queue, cpu_count
from concurrent.futures import ProcessPoolExecutor
import optuna
import random

from networks import *
from plot import *
from data import *


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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


def run_all_tasks(
    params,
    verbose_level=-1,
    plot_data=False,
    seed=0,
):
    (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    ) = params
    # Fix seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    net = Net()
    control_net = ControlNet()
    criterion = nn.CrossEntropyLoss()
    control_optimizer = torch.optim.Adam(control_net.parameters(), lr=float(control_lr))
    net_optimizer = torch.optim.Adam(net.parameters(), lr=float(learning_rate))

    all_losses = []
    task_performance = {}

    for task_id in tqdm(range(1, 4), desc="Tasks", disable=(verbose_level <= 0)):
        dataloader = get_dataloader(task_id)

        if plot_data:
            dataloader.dataset.plot()

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
                Above we trained the control network for the current state of the
                bio network. The control signal is at the beginning very strong
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
                
                Note: Bias
                From Sander:
                 > You can add biases, usually omitted in the equation because
                 > its implicit. (technically speaking the bias of a neuron is
                 > something added to make the flow of variance throughout the
                 > network correct, there are no explicit bias ‘parameters' in
                 > biological neurons (you do have biases but not trainable))

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
                                dw = r_pre * r_post * a
                                    - r_pre * r_post * a^*
                            
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
                        # net.set_control_signals(a_diff)
                        # TODO: Maybe we have to clip the a_diff?
                        signal_diff_layer1 = a_diff[:, : net.hidden_size]
                        aa = 1.2
                        signal_diff_layer1 = torch.clamp(
                            signal_diff_layer1, min=1.0 / aa, max=aa
                        )

                        signal_diff_layer2 = a_diff[:, net.hidden_size :]
                        signal_diff_layer2 = torch.clamp(
                            signal_diff_layer2, min=1.0 / aa, max=aa
                        )

                        # Layer 1
                        r_pre_hidden = net.flatten(batch_data)  # r_pre
                        r_post_hidden = net.hidden_activations(
                            net.layer1(r_pre_hidden)
                        )  # r_post * a

                        # dw = r_pre * r_post * a
                        foo = r_post_hidden * signal_diff_layer1
                        dw = foo.T @ r_pre_hidden
                        # dw = r_post_hidden.T @ r_pre_hidden
                        net.layer1.weight.grad = dw

                        # Layer 2
                        r_pre_output = r_post_hidden
                        r_post_output = net.output_activations(net.layer2(r_pre_output))

                        foo_out = r_post_output * signal_diff_layer2
                        dw = foo_out.T @ r_pre_output
                        # dw = r_post_output.T @ r_pre_output
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

        # for foo in net.parameters():
        #     print(foo.sum().item())
        # for foo in control_net.parameters():
        #     print(foo.sum().item())

    return params, task_performance


# Objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    num_epochs = trial.suggest_int("num_epochs", 100, 1500)
    inner_epochs = trial.suggest_int("inner_epochs", 10, 100)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
    control_lr = trial.suggest_float("control_lr", 1e-5, 1e-1)
    control_threshold = trial.suggest_float("control_threshold", 1e-11, 1e-6)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-4, 1e-1)

    # Run the model with the sampled parameters
    params = (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    )
    print("PARAMS = ", params)
    _, task_performance = run_all_tasks(params, verbose_level=0)

    # Evaluation metric: Average accuracy across tasks
    avg_accuracy = np.mean(
        [
            acc
            for task_results in task_performance.values()
            for acc in task_results.values()
        ]
    )

    # Goal is to maximize avg_accuracy
    return avg_accuracy


# Run the Optuna study
def run_optuna_study(num_trials, num_cpus):
    # Use SQLite as shared storage for parallel workers
    storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="hyperparameter_optimization",
        load_if_exists=True,
    )

    # Run optimization with parallel trials
    study.optimize(objective, n_trials=num_trials, n_jobs=num_cpus)

    # Print and return the best results
    print("Best parameters:", study.best_params)
    print("Best value (accuracy):", study.best_value)
    print("Best trial:", study.best_trial)
    return study


if __name__ == "__main__":
    # params1 = (
    #     355,
    #     96,
    #     0.031186578088439134,
    #     0.06858824436888418,
    #     9.310540712549254e-07,
    #     0.04721763083723491,
    # )
    # """
    #     Task 1 - Performance on Task 1: 52.00%
    #     Task 2 - Performance on Task 1: 26.00%
    #     Task 2 - Performance on Task 2: 48.00%
    #     Task 3 - Performance on Task 1: 46.00%
    #     Task 3 - Performance on Task 2: 50.00%
    #     Task 3 - Performance on Task 3: 52.00%
    # """
    # print("First")
    # run_all_tasks(params1, verbose_level=0)
    # params2 = (
    #     676,
    #     89,
    #     0.029868234667835978,
    #     0.09440888898158813,
    #     6.43637837977417e-08,
    #     0.009106209860749951,
    # )
    # print("second")
    # """
    #     Task 1 - Performance on Task 1: 50.00%
    #     Task 2 - Performance on Task 1: 36.00%
    #     Task 2 - Performance on Task 2: 48.00%
    #     Task 3 - Performance on Task 1: 52.00%
    #     Task 3 - Performance on Task 2: 38.00%
    #     Task 3 - Performance on Task 3: 54.00%
    # """
    # run_all_tasks(params2, verbose_level=0)
    # quit()
    # Configure the number of trials and CPUs
    num_trials = 50
    num_cpus = 1  # Adjust as needed

    # Run the study
    study = run_optuna_study(num_trials, num_cpus)

    # Show the best results
    print("Finished optimization. Best parameters:")
    print(study.best_params)
