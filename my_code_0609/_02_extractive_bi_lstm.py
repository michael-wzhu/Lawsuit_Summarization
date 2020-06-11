import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
	def __init__(self, hidden_size, vocab_size, class_num=1, layer_num=3):
		super(BiLSTM, self).__init__()
		self.H = hidden_size
		self.V = vocab_size
		self.C = class_num
		self.L = layer_num

		self.embed = nn.Embedding(self.V, self.H, padding_idx=0)
		# initialize embedding
		scope = np.sqrt(3.0 / self.embed.weight.size(1))
		nn.init.uniform_(self.embed.weight, -scope, scope)

		self.bilstm_word = nn.LSTM(
			input_size=self.H,
			hidden_size=self.H,
			num_layers=self.L,
			bidirectional=True,
			batch_first=True,
			bias=True)

		self.bilstm_sent = nn.LSTM(
			input_size=self.H*2,
			hidden_size=self.H*2,
			num_layers=self.L,
			bidirectional=True,
			batch_first=True,
			bias=True)

		self.out_projection = nn.Linear(in_features=self.H * 4, out_features=self.C, bias=True)
		# initialize linear
		scope = np.sqrt(6.0 / (self.H * 2 + self.C))
		nn.init.uniform_(self.out_projection.weight, -scope, scope)
		self.out_projection.bias.data.zero_()

		self.h0_word = torch.randn(self.L * 2, 1, self.H).float().cuda()
		self.c0_word = torch.randn(self.L * 2, 1, self.H).float().cuda()

		self.h0_sent = torch.randn(self.L * 2, 1, self.H*2).float().cuda()
		self.c0_sent = torch.randn(self.L * 2, 1, self.H*2).float().cuda()

		self.loss = nn.BCELoss()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x_ids, sents_len, words_len, y_mask):
		batch_size = len(x_ids)

		x_ = []
		x_length_max = max(sents_len)
		for i in range(batch_size):
			# 单词级别的编码
			x = self.embed(x_ids[i])  # [N, S, H]
			x_length = words_len[i]
			packed_x = pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
			hidden = (self.h0_word.repeat(1, x.size(0), 1), self.c0_word.repeat(1, x.size(0), 1))

			x, _ = self.bilstm_word(packed_x, hidden)
			x, _ = pad_packed_sequence(x, batch_first=True)  # [N, S, H*2]

			x_.append(torch.cat(
				[torch.mean(x, dim=1), torch.zeros([x_length_max-x.size(0), self.H*2], dtype=torch.float32).cuda()], dim=0)
			)

		# 句子级别的编码
		x = torch.stack(x_, dim=0)  # [N, S, H*2]
		packed_x = pack_padded_sequence(x, sents_len, batch_first=True, enforce_sorted=False)
		hidden = (self.h0_sent.repeat(1, x.size(0), 1), self.c0_sent.repeat(1, x.size(0), 1))

		x, _ = self.bilstm_sent(packed_x, hidden)
		x, _ = pad_packed_sequence(x, batch_first=True)  # [N, S, H*4]

		logits = self.sigmoid(self.out_projection(x))  # [N, S, 1]
		logits = logits.squeeze(-1) * y_mask

		return logits
