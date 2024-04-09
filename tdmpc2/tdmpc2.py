import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
#from common.world_model import WorldModel

from common import layers, math, init
from copy import deepcopy

class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
		self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
		
	def to(self, *args, **kwargs):
		"""
		Overriding `to` method to also move additional tensors to device.
		"""
		super().to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def track_q_grad(self, mode=True): # Similar
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		This method also enables/disables gradients for task embeddings.
		"""
		for p in self._Qs.parameters():
			p.requires_grad_(mode)

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with torch.no_grad():
			for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
				p_target.data.lerp_(p.data, self.cfg.tau)
	

	def encode(self, obs): # replaces h, similar
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a): # no more reward prediction, similar
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)
	
	def reward(self, z, a): # new
		"""
		Predicts instantaneous (single-step) reward.
		"""
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""

		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		action_dims = None # new

		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std # now also returns pi,log_pi, log_std

	def Q(self, z, a, return_type='min', target=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
			
		z = torch.cat([z, a], dim=-1) # Same
		out = (self._target_Qs if target else self._Qs)(z)

		if return_type == 'all':
			return out

		# bit different behaviour
		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
		return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		# std missing
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		# pi_optim missing
		self.model.eval()
		# target eval missing

		# this is new
		self.scale = RunningScale(cfg) 
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = self._get_discount(cfg.episode_length)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp): # same
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp): # Similar
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])
		# no target state loading

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		
		z = self.model.encode(obs)
		if self.cfg.mpc:
			a = self.plan(z, t0=t0, eval_mode=eval_mode)
		else:
			a = self.model.pi(z)[int(not eval_mode)][0]
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
			z = self.model.next(z, actions[t]) # reward and action are now split up
			G += discount * reward
			discount *= self.discount
		return G + discount * self.model.Q(z, self.model.pi(z)[1], return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z)[1]
				_z = self.model.next(_z, pi_actions[t])
			pi_actions[-1] = self.model.pi(_z)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)	

			# Compute elite actions # Same
			value = self._estimate_value(z, actions).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters # Same
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)

		# Select action # pretty much same
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		a, std = actions[0], std[0]
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a.clamp_(-1, 1) # previously no clamping here
		
	def update_pi(self, zs):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs)
		qs = self.model.Q(zs, pis, return_type='avg')
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# No enumerate anymore?
		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()

		# Exactly the same
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z)[1]
		return reward + self.discount * self.model.Q(next_z, pi, return_type='min', target=True)

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward = buffer.sample() # no more next_obses, idx, weights
	
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:])
			td_targets = self._td_target(next_z, reward)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		# no more std here
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0])
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t])
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, return_type='all')
		reward_preds = self.model.reward(_zs, action)
		
		# Compute losses # no more rho
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		#no more update to replay buffer

		# Update policy
		pi_loss = self.update_pi(zs.detach())

		# Update target Q-functions
		self.model.soft_update_target_Q() # replaces update_pi?

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value), # pi scale is new
		}
