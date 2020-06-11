from _01_data import *
from _02_extractive_bi_lstm import *
from tensorboardX import SummaryWriter
from torch import optim
from rouge import Rouge
import sys
sys.setrecursionlimit(10000000)


def train_extractive(out_path, hidden_size, learning_rate, max_epoch, batch_size):
	writer = SummaryWriter(log_dir='events/')
	train_data, vocab_size = load_data(mode="train")

	GLOBAL_STEP = 0

	model = BiLSTM(hidden_size, vocab_size).cuda()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for ep in range(max_epoch):
		metrics = []
		for batch_index in get_batch_index(len(train_data), batch_size):
		# for batch_index in get_batch_index(10, batch_size):
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

			p, r, f1, a = get_metrics(logits, y.int(), y_mask.int())
			metrics.append([p, r, f1, a])

			writer.add_scalar('training_loss', loss.item(), GLOBAL_STEP)
			GLOBAL_STEP += 1

			# backward
			loss.backward()
			optimizer.step()

		print('____training_metrics____EP.{}____'.format(ep))
		metrics = np.mean(np.array(metrics), axis=0).tolist()
		print("p\t{:.2f} | r\t{:.2f} | f1\t{:.2f} | acc\t{:.2f}".format(*metrics))
		print(''.format(ep))

		with open("../metrics/train_metrics.txt", 'a') as f:
			f.write("p\t{:.2f} | r\t{:.2f} | f1\t{:.2f} | acc\t{:.2f}\n".format(*metrics))
		valid_extractive(model, writer, batch_size, GLOBAL_STEP)
		torch.save(model, out_path + "hierBilstmExtrative-epoch_{}.ckpt".format(ep))


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

	return np.mean(np.array(all_scores), axis=0)


def valid_extractive(model, writer, batch_size, GLOBAL_STEP):
	test_data, _ = load_data(mode="test")

	summaries_record = []
	ground_truth = []
	metrics = []
	with torch.no_grad():
		losses = []
		for batch_index in get_batch_index(len(test_data), batch_size):
		# for batch_index in get_batch_index(60, batch_size):
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

			_, src_index = torch.topk(logits, 4, dim=-1)
			src_index = src_index.data.cpu().numpy().tolist()
			for i in range(batch_size):
				summary_i = ""
				summaries_i = []
				for j in src_index[i]:
					summary_i += sources[i][j] + ' '
					# summaries_i.append(summary_i.strip())
				summaries_record.append(summary_i)

			ground_truth.extend(summary)

	pred_path = '../output/{}_pred_y.txt'.format(GLOBAL_STEP)
	true_path = '../output/{}_true_y.txt'.format(GLOBAL_STEP)
	with open(pred_path, 'w', encoding='gbk') as f:
		f.writelines([s + '\n' for s in summaries_record])
	with open(true_path, 'w', encoding='gbk') as f:
		f.writelines([s + '\n' for s in ground_truth])

	# all_scores = get_rouge_scores(summaries_record, ground_truth)
	# print("all scores:", all_scores.tolist()[0])
	writer.add_scalar('validation_loss', sum(losses) / len(losses), GLOBAL_STEP)

	rouges = print_rouge_scores(pred_path, true_path)
	print('____validation_metrics____')
	metrics = np.mean(np.array(metrics), axis=0).tolist()
	print("p\t{:.2f} | r\t{:.2f} | f1\t{:.2f} | acc\t{:.2f}".format(*metrics))
	print("total\t{:.2f} | r1_f1\t{:.2f} | r2_f1\t{:.2f} | rl_f1\t{:.2f}".format(*rouges))

	print('')
	with open("../metrics/valid_metrics.txt", 'a') as f:
		f.write("p\t{:.2f} | r\t{:.2f} | f1\t{:.2f} | acc\t{:.2f}\n".format(*metrics))
	with open("../metrics/valid_rouge.txt", 'a') as f:
		f.write("total\t{:.2f} | r1_f1\t{:.2f} | r2_f1\t{:.2f} | rl_f1\t{:.2f}\n".format(*rouges))

if __name__ == "__main__":
	train_extractive('../output/', 256, 1e-5, 50, 32)
