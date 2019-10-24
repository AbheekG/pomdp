import torch
import torch.nn as nn
# from torchviz import make_dot

import models
import constants as ck


def train(model, max_cost=ck.D, horizon=ck.H, batch_size=128, epochs=100):
	losses = []
	# params_sums= []

	optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

	model.train()
	model = model.to(ck.device)

	max_cost = torch.zeros(batch_size, 1) + ck.D

	for epoch in range(epochs):
		# Randomly Uniformly select MDPs
		mdp = torch.multinomial(torch.ones(batch_size, ck.n_mdp)/ck.n_mdp, 1)
		assert mdp.size() == (batch_size, 1)

		# Init observations to state 0
		obs = nn.functional.one_hot(torch.zeros(batch_size).to(torch.int64),
			num_classes=ck.n_state).to(torch.float)
		assert obs.size() == (batch_size, ck.n_state)

		# Init belief to be uniform for each MDP.
		belief = torch.ones(batch_size, ck.n_mdp) / ck.n_mdp

		# Init cumulative cost to 0
		cum_cost = torch.zeros(batch_size, 1)

		# Init cell and hidden state of LSTM to 0.
		cell = torch.zeros(batch_size, models.BeliefLSTM.hidden)
		hidden = torch.zeros(batch_size, models.BeliefLSTM.hidden)

		# Forward pass
		for t in range(horizon):
			obs, belief, cum_cost, cell, hidden = model(mdp, max_cost, obs, belief, cum_cost, cell, hidden)

		loss = nn.NLLLoss()(torch.log(belief), mdp.flatten())

		# params_sum = []
		# for p in model.parameters():
		# 	print(p.name)
		# 	params_sum.append(p.data.sum())
		# params_sums.append(params_sum)

		# if epoch == 0:
		# 	dot = make_dot(loss, params=dict(model.named_parameters()))
		# 	dot.format = 'png'
		# 	dot.render()

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		
		print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

		# torch.save(model.state_dict(), 'model_%s.ckpt' % model.name)
	
	# print (torch.tensor(params_sums))

	return losses


if __name__ == "__main__":
	model = models.ClassifierRNN()
	losses = train(model, batch_size=512, epochs=1000)