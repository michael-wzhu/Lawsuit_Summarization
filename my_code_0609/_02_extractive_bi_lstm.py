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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df_loss = pd.read_csv('../output/infos/loss.csv', header=0)
df_metrics = pd.read_csv("../output/infos/metrics.csv", header=0)

loss_train = df_loss["train_loss"].values
loss_valid = df_loss["valid_loss"].values

train_p = df_metrics["train_p"].values
train_r = df_metrics["train_r"].values
train_f1 = df_metrics["train_f1"].values
train_acc = df_metrics["train_acc"].values

valid_p = df_metrics["valid_p"].values
valid_r = df_metrics["valid_r"].values
valid_f1 = df_metrics["valid_f1"].values
valid_acc = df_metrics["valid_acc"].values

val_zi = df_metrics["val_zi"].values
# val_r1_zi = df_metrics["val_r1_zi"].values
# val_r2_zi = df_metrics["val_r2_zi"].values
# val_rl_zi = df_metrics["val_rl_zi"].values
val_ci = df_metrics["val_ci"].values
# val_r1_ci = df_metrics["val_r1_ci"].values
# val_r2_ci = df_metrics["val_r2_ci"].values
# val_rl_ci = df_metrics["val_rl_ci"].values


num_epoch = len(loss_train)
steps = np.arange(1, num_epoch +1)

plt.subplot(2, 2, 1)
plt.plot(steps, loss_train, 'b--', label="train loss")
plt.plot(steps, loss_valid, 'r--', label="valid loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(steps, val_zi, 'kx-.', label="rouge based on single words")
plt.plot(steps, val_ci, 'ko:', label="rouge based on words")
plt.xlabel('epoch')
plt.ylabel('rouge score')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(steps, train_p, 'g--', label="train precision")
plt.plot(steps, train_r, 'y--', label="train recall")
plt.plot(steps, train_f1, 'bo--', label="train f1-score")
plt.plot(steps, train_acc, 'r--', label="train accuracy")
plt.xlabel('epoch')
plt.ylabel('train metrics')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(steps, valid_p, 'g--', label="valid precision")
plt.plot(steps, valid_r, 'y--', label="valid recall")
plt.plot(steps, valid_f1, 'bo--', label="valid f1-score")
plt.plot(steps, valid_acc, 'r--', label="valid accuracy")
plt.xlabel('epoch')
plt.ylabel('valid metrics')
plt.legend()

plt.show()
from _01_data import *
from _02_extractive_bi_lstm import *
from tensorboardX import SummaryWriter
from torch import optim
from rouge import Rouge
import csv
import sys
sys.setrecursionlimit(10000000)


def train_extractive(out_path, hidden_size, learning_rate, max_epoch, batch_size):
	writer = SummaryWriter(log_dir='events/')
	train_data, vocab_size = load_data(mode="train")

	GLOBAL_STEP = 0

	model = BiLSTM(hidden_size, vocab_size).cuda()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	with open("../output/infos/loss.csv", 'w') as f:
		csv_write = csv.writer(f)
		csv_head = ["train_loss", "valid_loss"]
		csv_write.writerow(csv_head)
	with open("../output/infos/metrics.csv", 'w') as f:
		csv_write = csv.writer(f)
		csv_head = [
			"train_p", "train_r", "train_f1", "train_acc",
			"valid_p", "valid_r", "valid_f1", "valid_acc",
			"val_zi", "val_r1_zi", "val_r2_zi", "val_rl_zi",
			"val_ci", "val_r1_ci", "val_r2_ci", "val_rl_ci",
		]
		csv_write.writerow(csv_head)

	for ep in range(max_epoch):

		record_metrics = []
		record_loss = []
		for batch_index in get_batch_index(len(train_data), batch_size):
		# for batch_index in get_batch_index(64, batch_size):
			batch_data = get_batch_data(train_data, batch_index)
			xs, _, _, sents_len, words_len, y, y_mask = process_batch_data(batch_data)

			y = torch.FloatTensor(y).cuda()
			y_mask = torch.FloatTensor(y_mask).cuda()

			optimizer.zero_grad()

			logits = model(
				[torch.LongTensor(x).cuda() for x in xs],
				sents_len,
				words_len,
				y_mask
			)  # [N, S]

			loss = model.loss(logits, y)

			record_loss.append(loss.item())
			p, r, f1, a = get_metrics(logits, y.int(), y_mask.int())
			record_metrics.append([p, r, f1, a])

			writer.add_scalar('training_loss', loss.item(), GLOBAL_STEP)
			GLOBAL_STEP += 1

			# backward
			loss.backward()
			optimizer.step()

		avg_metrics = np.mean(np.array(record_metrics), axis=0).tolist()
		avg_loss = np.mean(np.array(record_loss))

		# do validation every epoch
		valid_loss, valid_metric = valid_extractive(model, writer, batch_size, GLOBAL_STEP)

		# write records
		with open("../output/infos/loss.csv", 'a+', newline="") as f:
			csv_write = csv.writer(f)
			csv_write.writerow([avg_loss, valid_loss])
		with open("../output/infos/metrics.csv", 'a+', newline="") as f:
			csv_write = csv.writer(f)
			csv_write.writerow(avg_metrics + valid_metric)

		torch.save(model, out_path + "/models/hierBilstmExt-epoch_{}.ckpt".format(ep))


def get_metrics(logits, y, y_mask):
	eps = 1e-9
	y_pred = (logits > 0.5).int()
	TP = torch.sum((y_pred & y) * y_mask).float() + eps
	TN = torch.sum(((1-y_pred) & (1-y)) * y_mask).float() + eps
	FP = torch.sum((y_pred & (1-y)) * y_mask).float() + eps
	FN = torch.sum(((1-y_pred) & y) * y_mask).float() + eps

	p = TP.div(TP + FP)
	r = TP.div(TP + FN)
	f1 = (2 * p * r).div(p + r)
	a = (TP + TN).div(TP + TN + FP + FN)
	return p.item(), r.item(), f1.item(), a.item()


def print_rouge_scores(pred_path, true_path):
	get_rouge_scores = Rouge().get_scores
	with open(pred_path, 'r') as f:
		summaries = f.readlines()
	with open(true_path, 'r') as f:
		ground_truth = f.readlines()

	assert len(summaries) == len(ground_truth)

	all_scores = [] # 看不同的长度，那个rouge得分高
	for i in range(len(summaries)):

		# rouge_scores = get_rouge_scores(summaries[i][j], ground_truth[i])[0]
		hyps = ' '.join(list(summaries[i]))
		refs = ' '.join(list(ground_truth[i]))

		rouge_scores = get_rouge_scores(hyps, refs)[0]

		r1f = rouge_scores["rouge-1"]["f"]
		r2f = rouge_scores["rouge-2"]["f"]
		rlf = rouge_scores["rouge-l"]["f"]
		temp = r1f * 0.2 + r2f * 0.4 + rlf * 0.4
		all_scores.append([temp, r1f, r2f, rlf])

	rouge_based_on_zi = np.mean(np.array(all_scores), axis=0).tolist()

	# jieba 分词
	all_scores = [] # 看不同的长度，那个rouge得分高
	for i in range(len(summaries)):

		# rouge_scores = get_rouge_scores(summaries[i][j], ground_truth[i])[0]
		hyps = ' '.join([w for w in jieba.cut(summaries[i])])
		refs = ' '.join([w for w in jieba.cut(ground_truth[i])])
		rouge_scores = get_rouge_scores(hyps, refs)[0]

		r1f = rouge_scores["rouge-1"]["f"]
		r2f = rouge_scores["rouge-2"]["f"]
		rlf = rouge_scores["rouge-l"]["f"]
		temp = r1f * 0.2 + r2f * 0.4 + rlf * 0.4
		all_scores.append([temp, r1f, r2f, rlf])

	rouge_based_on_ci = np.mean(np.array(all_scores), axis=0).tolist()

	return rouge_based_on_zi + rouge_based_on_ci


def valid_extractive(model, writer, batch_size, GLOBAL_STEP):
	test_data, _ = load_data(mode="test")

	summaries_record = []
	ground_truth = []
	metrics = []
	with torch.no_grad():
		losses = []
		for batch_index in get_batch_index(len(test_data), batch_size):
		# for batch_index in get_batch_index(64, batch_size):
			batch_data = get_batch_data(test_data, batch_index)
			xs, sources, summary, sents_len, words_len, y, y_mask = process_batch_data(batch_data)

			y = torch.FloatTensor(y).cuda()
			y_mask = torch.FloatTensor(y_mask).cuda()

			logits = model(
				[torch.LongTensor(x).cuda() for x in xs],
				sents_len,
				words_len,
				y_mask
			)  # [N, S]

			loss = model.loss(logits, y)
			p, r, f1, a = get_metrics(logits, y.int(), y_mask.int())
			losses.append(loss.item())
			metrics.append([p, r, f1, a])

			_, src_index = torch.topk(logits, 5, dim=-1)
			src_index = src_index.data.cpu().numpy().tolist()
			for i in range(batch_size):
				summary_i = ""
				# summaries_i = []
				for j in src_index[i]:
					summary_i += sources[i][j] + ' '
					# summaries_i.append(summary_i.strip())
				summaries_record.append(summary_i)

			ground_truth.extend(summary)

	pred_path = '../output/preds/pred_y.txt'
	true_path = '../output/preds/true_y.txt'
	with open(pred_path, 'w', encoding='gbk') as f:
		f.writelines([s + '\n' for s in summaries_record])
	with open(true_path, 'w', encoding='gbk') as f:
		f.writelines([s + '\n' for s in ground_truth])

	avg_loss = sum(losses) / len(losses)
	writer.add_scalar('validation_loss', avg_loss, GLOBAL_STEP)

	rouges = print_rouge_scores(pred_path, true_path)
	metrics = np.mean(np.array(metrics), axis=0).tolist()

	return avg_loss, metrics + rouges

if __name__ == "__main__":
	train_extractive('../output/', 256, 1e-5, 50, 32)
import json
import re
import jieba
import pickle
import random
import numpy as np
import torch


input_path = "../sfzy_small/sfzy_small.json"
output_path = "/output/result.json"


def get_batch_index(num_sample, batch_size):
	num_batchs = np.ceil(num_sample / batch_size)
	i = 0
	indices = list(range(num_sample))
	random.shuffle(indices)

	while (i+1) * batch_size <= num_sample:
		yield indices[i*batch_size: min((i+1)*batch_size, num_sample)]
		i += 1


def get_batch_data(data, batch_index):
	return [data[i] for i in batch_index]


def process_batch_data(batch_data):
	summary = []
	labels = []
	sources = []
	sents = []
	sents_len = []
	words_len = []

	for d in batch_data:
		words_len.append([])
		summary.append(d[1])
		labels.append(d[4])
		sources.append(d[2])
		sents.append(d[3])
		sents_len.append(len(d[3]))

		for i, s in enumerate(d[3]):
			if len(s):
				words_len[-1].append(len(s))
			else:
				sents[-1].pop(i)
				labels[-1].pop(i)
				sources[-1].pop(i)

	batch_size = len(batch_data)

	with open("../input/dicts.pkl", "rb") as p:
		vocab, dict_w2i, dict_i2w = pickle.load(p)

	xs = []
	for i in range(batch_size):
		max_sent_len = max(words_len[i])
		x = np.zeros([sents_len[i], max_sent_len], dtype=np.int32)

		for j, sent in enumerate(sents[i]):
			for k, word in enumerate(sent):
				word_id = dict_w2i[word] if word in dict_w2i.keys() else dict_w2i["[UNK]"]
				x[j, k] = word_id
		xs.append(x)

	max_sent_num = max(sents_len)

	labels_ = np.zeros([batch_size, max_sent_num], dtype=np.float32)
	labels_mask = np.zeros([batch_size, max_sent_num], dtype=np.int32)
	for i, label_i in enumerate(labels):
		for j, _ in enumerate(label_i):
			labels_[i, j] = labels[i][j]
			labels_mask[i, j] = 1

	return xs, sources, summary, sents_len, words_len, labels_, labels_mask


def load_data(mode="train"):
	if mode == "train":
		with open('../input/train.pkl', 'rb') as p:
			data = pickle.load(p)
	else:
		with open('../input/test.pkl', 'rb') as p:
			data = pickle.load(p)
	with open("../input/dicts.pkl", "rb") as p:
		vocab, _, _ = pickle.load(p)

	return data, len(vocab)


def construct_dicts(words, vocab_size=30000, min_tf=1):
	dict_full = {}
	for w in words:
		if w in dict_full: dict_full[w] += 1
		else: dict_full[w] = 1

	vocab_full = sorted(dict_full.items(), key=lambda x: x[1], reverse=True)
	vocab = ["[PAD]", "[UNK]"]
	for v in vocab_full:
		if v[1] >= min_tf: vocab.append(v[0])
	vocab = vocab[:vocab_size]

	dict_i2w = dict(zip(list(range(len(vocab))), vocab))
	dict_w2i = dict(zip(vocab, list(range(len(vocab)))))

	with open("../input/dicts.pkl", "wb") as p:
		pickle.dump((vocab, dict_w2i, dict_i2w), p)


def processSourceText(text, all_words):
	# 逐个读取列表元素
	sents = []
	labels = []
	src = []

	for t in text:
		sent = t["sentence"].replace('\u3000', '').replace('\x20', '').replace('\xa0', '').replace(' ', '')
		src.append(sent)
		sent = re.sub('[0-9a-zA-Z]+', ' # ', sent)
		sents.append(sent)
		labels.append(t["label"])

	sents_cut = []
	for i in range(len(sents)):
		sent_cut = [word for word in jieba.cut(sents[i], cut_all=False) if word != ' ']
		sents_cut.append(sent_cut)
		all_words.extend(sent_cut)

	return src, sents_cut, labels


if __name__ == "__main__":
	with open(input_path, 'r', encoding="utf8") as f:
		processed_data = []
		all_words = []
		for line in f:
			data = json.loads(line)
			id = data.get("id")  # a string
			text = data.get("text")  # list of dicts
			summary = data.get("summary")  # a string

			sents, sents_cut, labels = processSourceText(text, all_words)

			processed_data.append([id, summary, sents, sents_cut, labels])

		random.shuffle(processed_data)
		train_size = int(len(processed_data) * 0.95)

	with open('../input/train.pkl', 'wb') as p:
		pickle.dump(processed_data[:train_size], p)
	with open('../input/test.pkl', 'wb') as p:
		pickle.dump(processed_data[train_size:], p)

	# with open('../input/train.pkl', 'rb') as p:
	# 	train_data = pickle.load(p)
	# with open('../input/test.pkl', 'rb') as p:
	# 	test_data = pickle.load(p)

	construct_dicts(all_words)