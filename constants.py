import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_mdp = 2
n_state = 3
n_action = 3

# C(i, j) = C(State, Action)
C = torch.zeros(n_state, n_action)
C = torch.tensor([
	[2.0, 5.0, 0],
	[6.0, 4.0, 0],
	[7.0, 7.0, 0]])
assert C.shape == (n_state, n_action)

# T(MDP, Action, State, Next-State)
T = torch.zeros(n_mdp, n_action, n_state, n_state)
T[0][0] = torch.tensor([
	[0.8, 0.2,   0],
	[0.7, 0.2, 0.1],
	[  0,   0,   1]])
T[0][1] = torch.tensor([
	[0.6, 0.4,   0],
	[0.2, 0.4, 0.4],
	[  0,   0,   1]])
T[0][2] = torch.tensor([
	[0.5, 0.5,   0],
	[0.1, 0.6, 0.3],
	[  0,   0,   1]])

T[1][0] = torch.tensor([
	[0.6, 0.4,   0],
	[0.1, 0.5, 0.4],
	[  0,   0,   1]])
T[1][1] = torch.tensor([
	[0.9, 0.1,   0],
	[0.8, 0.1, 0.1],
	[  0,   0,   1]])
T[1][2] = torch.tensor([
	[0.3, 0.7,   0],
	[0.1, 0.3, 0.6],
	[  0,   0,   1]])
assert T.shape == (n_mdp, n_action, n_state, n_state)

# Initial belief.
B = torch.tensor([0.5, 0.5])
assert B.shape == (n_mdp,)

# Cost constraint
D = 10

# Default horizon length.
H = 5

# Threshold probabilities
ld = {
	"a": torch.tensor([0.8, 0.7]),
	"b": torch.tensor([0.9, 0.8]),
	"c": torch.tensor([0.95, 0.9])
}

# Without reaching state 3. TODO