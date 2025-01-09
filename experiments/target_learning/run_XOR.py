from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import optuna
import random
import sqlite3
from torchviz import make_dot
from nn.Net import Net
from nn.ControlNet import ControlNet
from dataloaders.ContinualLearningDataset import get_dataloader


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

    for task_id in range(1, 4):
        dataloader = get_dataloader(task_id)

        if plot_data:
            dataloader.dataset.plot()

        task_losses = []

        pbar = tqdm(
            range(num_epochs),
            desc=f"Task {task_id} Epochs",
            leave=False,
            disable=(verbose_level <= 0),
        )

        for epoch in pbar:
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
                    disable=(True or verbose_level <= 0),
                ):
                    control_optimizer.zero_grad()

                    # Question: Why do we get the target signal for these
                    # activities? Shouldn't he take the target signal wtt to the
                    # inputs of ReLu layers?
                    control_signals = control_net(current_activities)

                    # ATTENTION: This is implemented by
                    #      self.control_signals * torch.relu(x)
                    # whereas control_signals comes from control_net and
                    # x comes from net.
                    #
                    # !! The multiplication connects the computational graph
                    # of control_net with the computational graph of net.
                    #
                    # If we do total_control_loss.backward() later, it
                    # will backpropagate through this connected comp. graph.
                    # This means it will update weight.grad for both layer!
                    #
                    # Now this shouldn't be a problem because we overwrite
                    # net's weight.grad later on manually.
                    #
                    # The big question is: Do the weight.grad change due to the
                    # changed comp. graph?
                    net.set_control_signals(control_signals)

                    # params = {
                    #     **dict(net.named_parameters()),
                    #     **dict(control_net.named_parameters()),
                    # }

                    if verbose_level >= 2:
                        print("Control signals", control_signals.mean())

                    output = net(batch_data)  # net is excluded from the graph
                    control_loss = criterion(output, batch_labels)
                    # graph = make_dot(control_loss, params=params)
                    # graph.render("combined_computational_graph", format="pdf")
                    l1_reg = (
                        l1_lambda
                        * (net(batch_data) - batch_labels).abs().sum(dim=1).mean()
                    )

                    total_control_loss = control_loss + l1_reg

                    total_control_loss.backward()
                    control_optimizer.step()

                    if abs(prev_loss - total_control_loss.item()) < control_threshold:
                        if verbose_level >= 2:
                            print("  Converged at epoch", inner_epoch)
                        break

                    prev_loss = total_control_loss.item()

                    epoch_losses.append(control_loss.item())

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
                        Theory:
                            In the paper we see that we have a multiplicative
                            neuron model: (eq. 24)

                                varphi(z,a) = phi(z) * a                    (24)
                            
                                whereas phi is our activation function (relu) and
                                a is our control signal.
                            
                            It also gives us the weight update formula (eq. 25):

                                dw = r_pre * r_post * (a - a^*)             (25)
                                   = r_pre * (phi(z) * a) * (a - a^*)

                            For a presynaptic neuron i and postsynaptic neuron j
                            we get:

                                dw_ij = x_i * [ phi( sum_i(w_ij * x_i) ) * a_j ] * (a_j - 1.0)   

                            Note that this is the formula to update one specific
                            weight of one connection between two neurons. The
                            first neuron is the presynaptic neuron and the second
                            neuron is the postsynaptic neuron.

                            - r_pre is just the output of the first neuron (after activation)
                            - r_post is the output of the second neuron  (after activation)
                            - a^* is 1.0 because eq. 32 tells us 
                              a(t) = 1 + alpha * f(t) and we want to end up with
                              no target signal hence f(t) = 0 => a^* = 1.0
                        """

                        # a.shape is [batch_size, hidden_size + output_size]
                        a1 = control_signals[:, : net.hidden_size]
                        a2 = control_signals[:, net.hidden_size :]
                        # Question: How to figure out baseline?
                        # A: Sander said just take 1.0
                        baseline_a1 = torch.ones_like(a1)
                        baseline_a2 = torch.ones_like(a2)
                        a1_diff = a1 - baseline_a1
                        a2_diff = a2 - baseline_a2

                        # Note: For the layer1 and layer2 weight updates we could
                        # move part of the inner loop to the outer loop since it
                        # doesn't depend on j. I just didn't do it because I figured
                        # it might be more confusing if I did.

                        #
                        # LAYER 1 WEIGHT UPDATE
                        #

                        # x.shape is [batch_size, input_size]
                        x = net.flatten(batch_data)

                        # phi.shape is [batch_size, hidden_size]
                        phi = net.hidden_activations(net.layer1(x))

                        # Question: Do we actually use the control signal
                        # correctly? Should we maybe use
                        # set_control_signal()?

                        # Loop over post-synaptic neurons (output neurons of layer 1)
                        for i in range(net.hidden_size):
                            # Question: Is r_post = phi*a really true?

                            # The post synaptic neuron j has output phi_i
                            r_post = (
                                phi[:, i] * a1[:, i]
                            )  # r_post.shape is [batch_size]

                            # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                            for j in range(net.input_size):
                                # Post synaptic neuron i has presynaptic signal j
                                r_pre_j = x[:, j]  # r_pre.shape is [batch_size]

                                dw_ij = (
                                    r_pre_j * r_post * a1_diff[:, i]
                                )  # dw_i.shape is [batch_size]

                                # We take the mean because we have a batch!
                                # Note: We set the gradient of the weight because later on
                                # we use an optimizer to update the weight.
                                net.layer1.weight.grad[i, j] = dw_ij.mean()

                        #
                        # LAYER 2 WEIGHT UPDATE
                        #
                        x = net.hidden_activations(
                            net.layer1(net.flatten(batch_data))
                        )  # x.shape is [batch_size, hidden_size]

                        # Question: phi here isn't the same as net(data) because
                        # after the ReLu there's also a softmax. Above in
                        # current_activities they use net(data) as the activities.
                        # So maybe I have to do the same here?

                        phi = net.output_activations(
                            net.layer2(x)
                        )  # phi.shape is [batch_size, output_size]

                        # Question: Do we need the above phi or this one?
                        # The difference is, the later is just a softmax
                        # applied to the former, which is the real output of
                        # net.
                        # A: I doubt it's correct.
                        # phi = net(batch_data)

                        # Loop over post-synaptic neurons (output neurons of layer 2)
                        for i in range(net.output_size):
                            # The post synaptic neuron j has output phi_i
                            r_post = (
                                phi[:, i] * a2[:, i]
                            )  # r_post.shape is [batch_size]

                            # Loop over presynaptic signals (the input signals for the i-th post-synaptic neuron)
                            for j in range(net.hidden_size):
                                # Post synaptic neuron i has presynaptic signal j
                                r_pre_i = x[:, j]  # r_pre.shape is [batch_size]

                                dw_i = (
                                    r_pre_i * r_post * a2_diff[:, i]
                                )  # dw_i.shape is [batch_size]

                                # We take the mean because we have a batch!
                                net.layer2.weight.grad[i, j] = dw_i.mean()

                        net_optimizer.step()

            avg_epoch_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            )
            task_losses.append(avg_epoch_loss)
            if epoch % 1 == 0 and verbose_level >= 1:
                pbar.set_postfix(avg_epoch_loss=avg_epoch_loss)
                # print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")

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
    num_epochs = trial.suggest_int("num_epochs", 10, 200)
    inner_epochs = trial.suggest_int("inner_epochs", 10, 200)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    control_lr = trial.suggest_float("control_lr", 1e-6, 1e-1, log=True)
    control_threshold = trial.suggest_float("control_threshold", 1e-8, 1e-3, log=True)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-3, 2e-1, log=True)

    # Run the model with the sampled parameters
    params = (
        num_epochs,
        inner_epochs,
        learning_rate,
        control_lr,
        control_threshold,
        l1_lambda,
    )
    _, task_performance = run_all_tasks(params, verbose_level=-1)

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
    storage = "sqlite:///study.db?check_same_thread=False&pool_size=20&max_overflow=48"
    conn = sqlite3.connect("optuna_study.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()
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


def avg(data):
    values = [value for subdict in data.values() for value in subdict.values()]
    print("Avg Perf.: ", sum(values) / len(values))


if __name__ == "__main__":
    # You can run the XOR experiment with a specifi set of hyperparams:

    # params_75_to_80 should give between 75% and 80%
    params_75_to_80 = {
        "num_epochs": 86,
        "inner_epochs": 50,
        "learning_rate": 5.740229923548619e-06,
        "control_lr": 0.00017419201936184192,
        "control_threshold": 0.0005192301553601519,
        "l1_lambda": 0.007533898129929821,
    }

    _, perf = run_all_tasks(params_75_to_80.values(), verbose_level=1, plot_data=False)
    avg(perf)

    # # Remove
    quit()

    # Or you can start a hyperparameter optimization study with Optuna:
    # Configure the number of trials and CPUs
    num_trials = 1000
    # Adjust as needed, max is 48 (change storage string abvoe to increase max)
    num_cpus = 8

    # Run the study
    study = run_optuna_study(num_trials, num_cpus)

    # Show the best results
    print("Finished optimization. Best parameters:")
    print(study.best_params)
