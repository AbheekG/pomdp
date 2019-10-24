import torch
import torch.nn as nn

import constants as ck


class ActionNN(nn.Module):
	"""
	The neural network to decide action, takes input:
	- observation
	- belief
	- cumulative cost
	- (maybe) max-cost

	Output:
	- action
	"""
	def __init__(self):
		super(ActionNN, self).__init__()

		# Params	
		d_in1 = ck.n_state  # The observation
		d_in2 = ck.n_mdp  # The belief
		d_in3 = 2  # The current and total cost.

		d_hidden11 = 5
		d_hidden12 = 5
		d_hidden13 = 5
		d_hidden1 = d_hidden11 + d_hidden12 + d_hidden13

		d_hidden2 = 32
		d_hidden3 = 32

		d_out = ck.n_action
		drop = 0.3
		
		self.drop11 = nn.Dropout(drop)
		self.layer11 = nn.Linear(d_in1, d_hidden11)
		self.nl11 = nn.ReLU()

		self.drop12 = nn.Dropout(drop)
		self.layer12 = nn.Linear(d_in2, d_hidden12)
		self.nl12 = nn.ReLU()

		self.drop13 = nn.Dropout(drop)
		self.layer13 = nn.Linear(d_in3, d_hidden13)
		self.nl13 = nn.ReLU()

		self.drop2 = nn.Dropout(drop)
		self.layer2 = nn.Linear(d_hidden1, d_hidden2)
		self.nl2 = nn.ReLU()

		self.drop3 = nn.Dropout(drop)
		self.layer3 = nn.Linear(d_hidden2, d_hidden3)
		self.nl3 = nn.ReLU()

		self.drop4 = nn.Dropout(drop)
		self.layer4 = nn.Linear(d_hidden3, d_out)
		self.nl4 = nn.Softmax(dim=-1)

	def forward(self, obs, bel, cost, max_cost):
		y_pred1 = self.nl11(self.layer11(self.drop11(obs)))
		y_pred2 = self.nl12(self.layer12(self.drop12(bel)))
		y_pred3 = self.nl13(self.layer13(self.drop13( torch.cat([cost, max_cost], -1) )))

		y_pred = torch.cat([y_pred1, y_pred2, y_pred3], axis=-1)
		y_pred = self.nl2(self.layer2(self.drop2(y_pred)))
		y_pred = self.nl3(self.layer3(self.drop3(y_pred))) + y_pred
		y_pred = self.nl4(self.layer4(self.drop4(y_pred)))  # Softmax in last layer.
		return y_pred


def env(mdp, obs, action):
	"""
	The environment layer. Takes input:
	- mdp
	- action
	- current obs

	Returns:
	- cost
	- new obs (transition)
	"""
	assert mdp.size()[0] == obs.size()[0] == action.size()[0]  # Batch size
	batch_size = mdp.size()[0]

	mdp = mdp.flatten()
	action = torch.multinomial(action, 1).flatten()
	obs = torch.multinomial(obs, 1).flatten()

	cost = ck.C[obs, action].view(batch_size, 1)
	new_obs = nn.functional.one_hot(torch.multinomial(ck.T[mdp, action, obs], 1),
		num_classes=ck.n_state).to(torch.float).view(batch_size, ck.n_state)

	return cost, new_obs


class EnvTC(nn.Module):
	"""
	Learns the transition and cost matrices. Our learning module tries
	to align this with the real 'env' function.
	Takes input:
	- mdp
	- action
	- current obs

	Returns:
	- cost
	- new obs (transition)
	"""
	def __init__(self):
		super(EnvTC, self).__init__()

		# C(i, j) = C(State, Action)
		param_C = ck.C.new().normal_(0, 1)
		param_C = nn.Sigmoid()(param_C)
		self.T = nn.Parameter(param_C)

		# T(MDP, Action, State, Next-State)
		param_T = ck.T.new().normal_(0, 1)  # Getting new tensor same type as T
		param_T = nn.Softmax(dim=-1)(param_T)  # Normalize to be prob. dist over next state
		self.T = nn.Parameter(param_T)

	def forward(self, mdp, obs, action):
		assert mdp.size()[0] == action.size()[0] == obs.size()[0]  # Batch size.
		batch_size = mdp.size()[0]

		batch_C = self.C.view(1, ck.n_state, ck.n_action)  # C(i, j) = C(State, Action)
		cost = ((batch_C * action.view(batch_size, 1, ck.n_action)).sum(axis=2) *
			obs).sum(axis=1).view(batch_size, 1)

		batch_T = self.T[mdp]  # T(MDP/idx in batch, Action, State, Next-State)
		assert batch_T.size() == (batch_size, ck.n_action, ck.n_state, ck.n_state)
		new_obs = ((batch_T * action.view(batch_size, ck.n_action, 1, 1)).sum(axis=1) *
			obs.view(batch_size, ck.n_state, 1)).sum(axis=1)
		assert new_obs.size() == (batch_size, ck.n_state)
		
		return cost, new_obs


class PreBeliefNN(nn.Module):
	"""
	The nn to convert action and new obs to input for LSTM. Takes input:
	- action
	- new obs

	Output:
	- input to LSTM
	"""
	# Shared param
	out = 32

	def __init__(self):
		super(PreBeliefNN, self).__init__()

		# Params
		d_in1 = ck.n_action
		d_in2 = ck.n_state
		
		d_hidden11 = 16
		d_hidden12 = 16
		d_hidden1 = d_hidden11 + d_hidden12
		d_hidden2 = 32
		
		d_out = PreBeliefNN.out
		drop = 0.3
		
		# Layers
		self.drop11 = nn.Dropout(drop)
		self.layer11 = nn.Linear(d_in1, d_hidden11)
		self.nl11 = nn.ReLU()

		self.drop12 = nn.Dropout(drop)
		self.layer12 = nn.Linear(d_in2, d_hidden12)
		self.nl12 = nn.ReLU()

		self.drop2 = nn.Dropout(drop)
		self.layer2 = nn.Linear(d_hidden1, d_hidden2)
		self.nl2 = nn.ReLU()

		self.drop3 = nn.Dropout(drop)
		self.layer3 = nn.Linear(d_hidden2, d_out)
		self.nl3 = nn.ReLU()

	def forward(self, action, obs):
		y_pred1 = self.nl11(self.layer11(self.drop11(action)))
		y_pred2 = self.nl12(self.layer12(self.drop12(obs)))

		y_pred = torch.cat([y_pred1, y_pred2], axis=-1)
		y_pred = self.nl2(self.layer2(self.drop2(y_pred))) + y_pred
		y_pred = self.nl3(self.layer3(self.drop3(y_pred)))
		return y_pred


class BeliefLSTM(nn.Module):
	"""
	The belief LSTM, takes input:
	- the old cell state
	- the old hidden state
	- (maybe) belief
	- (maybe) previous observation
	- lstm_input from preBelief

	Output:
	- cell state
	- hidden state
	"""
	hidden = 10
	def __init__(self):
		super(BeliefLSTM, self).__init__()

		# Params
		d_in = PreBeliefNN.out
		d_hidden = BeliefLSTM.hidden
		
		# Layers
		self.layer_f = nn.Linear(d_in + d_hidden, d_hidden)
		self.nl_f = nn.Sigmoid()
		self.layer_i = nn.Linear(d_in + d_hidden, d_hidden)
		self.nl_i = nn.Sigmoid()
		self.layer_o = nn.Linear(d_in + d_hidden, d_hidden)
		self.nl_o = nn.Sigmoid()
		self.layer_c = nn.Linear(d_in + d_hidden, d_hidden)
		self.nl_c = nn.Tanh()
		self.nl_h = nn.Tanh()

	def forward(self, c_prev, h_prev, x):
		xh_prev = torch.cat([x, h_prev], axis=-1)
		f = self.nl_f(self.layer_f(xh_prev))
		i = self.nl_i(self.layer_i(xh_prev))
		o = self.nl_o(self.layer_o(xh_prev))
		c = f*c_prev + i*self.nl_c(self.layer_c(xh_prev))
		h = o*self.nl_h(c)
		return c, h


class PostBeliefNN(nn.Module):
	"""
	The nn to convert hidden state to belief. Takes input:
	- hidden state of LSTM

	Output:
	- belief
	"""
	def __init__(self):
		super(PostBeliefNN, self).__init__()

		# Params
		d_in = BeliefLSTM.hidden
		d_hidden = 10
		d_out = ck.n_mdp
		drop = 0.3
		
		# Layers
		self.drop1 = nn.Dropout(drop)
		self.layer1 = nn.Linear(d_in, d_hidden)
		self.nl1 = nn.ReLU()

		self.drop2 = nn.Dropout(drop)
		self.layer2 = nn.Linear(d_hidden, d_out)
		self.nl2 = nn.Softmax(dim=-1)

	def forward(self, x):
		y_pred = self.nl1(self.layer1(self.drop1(x)))
		y_pred = self.nl2(self.layer2(self.drop2(y_pred)))
		return y_pred


class ClassifierRNN(nn.Module):
	"""
	Combines all the parts to form the complete cycle.
	"""
	def __init__(self):
		super(ClassifierRNN, self).__init__()

		# Action. Input: obs, bel, cost, max_cost. Output: action
		self.action_nn = ActionNN()
		# Env. Input: mdp, obs, action. Output: new_obs
		self.env_ours = EnvTC()
		self.env_real = env
		# PreBelief. Input: action, new_obs. Output: LSTM_input
		self.pre_belief_nn = PreBeliefNN()
		# Belief. Input: cell state, hidden, LSTM_input. Output: new_cell, new_hidden state
		self.belief_lstm = BeliefLSTM()
		# PostBelief. Input: new hidden. Output: new belief.
		self.post_belief_nn = PostBeliefNN()

	def forward(self, mdp, max_cost, obs, belief, cum_cost, cell, hidden):
		action = self.action_nn(obs, belief, cum_cost, max_cost)
		cost, new_obs = self.env_real(mdp, obs, action)
		lstm_in = self.pre_belief_nn(action, new_obs)
		new_cell, new_hidden = self.belief_lstm(cell, hidden, lstm_in)
		new_belief = self.post_belief_nn(new_hidden)

		new_cum_cost = cum_cost + cost

		return new_obs, new_belief, new_cum_cost, new_cell, new_hidden
