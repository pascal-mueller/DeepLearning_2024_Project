# Setup
0. python -V should be 3.11.6 (This only matters later for reproducibility)
1. python -m venv venv
2. source venv/bin/activate 
3. pip install -r requirements.txt
3. python run_XOR.py

# Run
E.g. do `python -m experiments.target_learning.run_XOR`

Note: Don't you `python <path>`.

We could make a `main.py` or something at the root level.

Results are stored in `./results/<exp_name>/<run_name>`

## Examples:

Run backprop XOR: `./run.py --name bp_full_XOR --run_name "test_run_0"`
Run param search for backprop XOR
- `./run.py --name bp_full_XOR --paramsearch --run_name param_search_0 --num_cores 8 --num_trials 8`

Run target learning XOR: `./run.py --name tl_full_XOR --run_name "test_run_0"`
Run param search for target learning XOR
- `./run.py --name tl_full_XOR --paramsearch --run_name param_search_0 --num_cores 8 --num_trials 8`
# Attention
Note 1:
This currently does not pass 100% accuracy for all thet tasks. I have no idea
why. I reread the paper several times and I really think it's implemented
correctly now. I even made the implementation very naive to make sure I don't do
any mistakes when it comes to mathematical operations etc.

Now there could be issues like vanishign gradients, exploding gradients and
other "details" like that which I didn't caught yet.

Note 2:
The results might not be reproducable. Just get it working locally and then we
can care about reproducability. Last time I added reproducability i.e. getting
rid of anything non-deterministic resp. fixed any "randomness", it somehow
fucked up my accuracy.

# Euler
We have a student cluster but I didn't check it out yet.

To set up environment:
1. ssh onto euler
2. module load stack/.2024-06-silent  gcc/12.2.0
3. module load python/3.11.6

Interactive shell: srun -n 8 --time=4:00:00 --pty bash

You can also submit the job directly, I never do that though so no idea how.
See wiki. I suggest starting an interactive shell with tmux. if you use tmux,
note down the login node so you can ssh onto it later on since the tmux session
is "login node specific".

# Commands used in paper
- `python -m experiments.target_learning.MNIST` checks our exp. setup on the whole MNIST dataset.