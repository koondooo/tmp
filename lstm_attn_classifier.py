#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAttnClassifier(nn.Module):
    def __init__(self, num_words, num_classes, hidden_size):
        super(LSTMAttnClassifier, self).__init__()
        input_size = num_words + 1 # including pad
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        # embedding
        embedded = self.embedding(pad_sequence(input, batch_first=True))

        # LSTM
        input_lengths = [x.size()[0] for x in input]
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        lstm_last_out = torch.cat([v[size-1].unsqueeze(dim=0) for v, size in zip(*pad_packed_sequence(lstm_out, batch_first=True))])

        # attension
        lstm_out_expanded = lstm_last_out.unsqueeze(dim=1).expand(-1, embedded.size()[1], -1)
        attn_scores = F.cosine_similarity(embedded, lstm_out_expanded, dim=2)
        attn_scores[attn_scores==0] = float("-inf") # set -inf at empty cells to calculate softmax
        attn_weights = F.softmax(attn_scores, dim=1)

        # apply attension
        attn_applied = torch.mul(embedded,
                attn_weights.unsqueeze(dim=2) #.expand(-1, -1, embedded.size()[2]) # expand is unnecessary because the shapes of the tensors are 'broadcastable'
                )
        hidden = torch.sum(attn_applied, dim=1)

        # output
        output = self.linear(hidden)

        return output

def make_XY(data):
    inputs, labels = [], []
    # sort for pad_sequence
    for d in sorted(data, key=lambda d: len(d["x"]), reverse=True):
        inputs.append(torch.tensor(d["x"]).to(device))
        labels.append(d["y"])
    labels = torch.tensor(labels).to(device)
    return inputs, labels

if __name__ == "__main__":
    num_words = 10
    num_classes = 4
    hidden_size = 20
    model = LSTMAttnClassifier(num_words, num_classes, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # train
    # x must start from 1 because 0 is used for pad
    # y must start from 0
    data_train = [
        {"x": [1, 2, 3],    "y": 0},
        {"x": [4, 5, 6, 7], "y": 1},
        {"x": [8],          "y": 2},
        {"x": [9, 10],      "y": 3}
    ]
    inputs, labels = make_XY(data_train)

    for epoch in range(0, 100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
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
