# Setup
0. python -V should be 3.11.6 (This only matters later for reproducibility)
1. python -m venv venv
2. source venv/bin/activate 
3. pip install -r requirements.txt
3. python run_XOR.py

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