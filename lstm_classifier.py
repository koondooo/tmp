#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMClassifier(nn.Module):
	def __init__(self, input_size, num_classes, hidden_size):
		super(LSTMClassifier, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
		self.lstm = nn.LSTM(hidden_size, hidden_size)
		self.linear = nn.Linear(hidden_size, num_classes)

	def forward(self, input):
		embedded = self.embedding(pad_sequence(input))

		input_lengths = [x.size()[0] for x in input]
		packed = pack_padded_sequence(embedded, input_lengths)
		output, hidden_cell_states = self.lstm(packed)
		hidden_state, cell_state = hidden_cell_states
		hidden = hidden_state[0] # this index '0' depends on the LSTM setting (num_layers and bidirectional)

		output = self.linear(hidden)

		return F.softmax(output, dim=1)

def make_XY(data):
	inputs, labels = [], []
	# sort for pad_sequence
	for d in sorted(data, key=lambda d: len(d["x"]), reverse=True):
		inputs.append(torch.tensor(d["x"]).to(device))
		labels.append(d["y"])
	labels = torch.tensor(labels).to(device)
	return inputs, labels

if __name__ == "__main__":
	num_words = 11 # including 0 for pad
	num_classes = 4
	hidden_size = 20
	model = LSTMClassifier(num_words, num_classes, hidden_size).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	# train
	data_train = [
		{"x": [1, 2, 3],    "y": 0},
		{"x": [4, 5, 6, 7], "y": 1},
		{"x": [8],          "y": 2},
		{"x": [9, 10],      "y": 3}
	]
	inputs, labels = make_XY(data_train)

	for epoch in range(0, 10000):
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		if epoch % 100 == 99:
			print("loss is %.3f" % loss.item())

	# test
	data_test = [
		{"x": [1, 2, 3],    "y": 0},
		{"x": [1, 3],       "y": 0},
		{"x": [1, 2, 3, 4], "y": 0},
		{"x": [4, 5, 6, 7], "y": 1},
		{"x": [4, 7],       "y": 1},
		{"x": [1, 4, 5, 6], "y": 1},
		{"x": [8],          "y": 2},
		{"x": [9, 10],      "y": 3},
		{"x": [9],          "y": 3}
	]
	inputs, labels = make_XY(data_test)

	with torch.no_grad():
		outputs = model(inputs)
		_, predicted = torch.max(outputs, 1)
		for predict, label in zip(predicted, labels):
			print("%d is predicted to %d" % (label, predict))
