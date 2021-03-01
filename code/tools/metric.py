import torch
import ot
import math


def compute_missing_rate(mask):
	missing_rate = 1 - torch.round(mask).sum().item() / torch.numel(mask)
	return missing_rate


def compute_first_wasserstein_distance(p, q):
	num_sample_p = p.size(0)
	num_sample_q = q.size(0)
	p = p.view(num_sample_p, -1)
	q = q.view(num_sample_q, -1)
	p_squared = (p * p).sum(dim=1).resize_(num_sample_p, 1)
	q_squared = (q * q).sum(dim=1).resize_(num_sample_q, 1)
	m_euclidean = torch.zeros(num_sample_p, num_sample_q)
	m_euclidean.copy_(p_squared.expand(num_sample_p, num_sample_q) + q_squared.expand(num_sample_q, num_sample_p).transpose(0, 1) - 2 * torch.mm(p, q.transpose(0, 1)))
	m_euclidean = m_euclidean.abs().sqrt()
	del p, q, p_squared, q_squared
	first_wasserstein_distance = ot.emd2([], [], m_euclidean.numpy())
	return first_wasserstein_distance


def compute_mean_absolute_error(real_data, imputed_data, real_mask):
	mean_abosulte_error = (real_data - imputed_data).abs_().sum().item() / (torch.numel(real_mask) - real_mask.sum().item())
	return mean_abosulte_error


def compute_root_mean_square_error(real_data, imputed_data, real_mask):
	root_mean_square_error = math.sqrt(torch.square(real_data - imputed_data).sum().item() / (torch.numel(real_mask) - real_mask.sum().item()))
	return root_mean_square_error
