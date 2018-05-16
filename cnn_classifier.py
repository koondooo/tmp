#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    def __init__(self, num_words, embedding_size, input_channel, output_channel, filter_heights, max_sentence_len, num_classes):
        super(CNNClassifier, self).__init__()
        self.filter_heights = filter_heights
        self.max_sentence_len = max_sentence_len

        input_size = num_words + 1 # including pad
        self.embeddings = [nn.Embedding(input_size, embedding_size, padding_idx=0) for _ in range(0, input_channel)]
        self.add_modules("embedding", self.embeddings)

        self.convs = [nn.Conv2d(input_channel, output_channel, (filter_height, embedding_size), padding=0) for filter_height in filter_heights]
        self.add_modules("conv", self.convs)

        self.linear1 = nn.Linear(output_channel * len(self.convs), output_channel * len(self.convs))
        self.linear2 = nn.Linear(output_channel * len(self.convs), num_classes)

    def forward(self, input, training=False):
        embeddeds = [embedding(input) for embedding in self.embeddings]
        embeddeds = torch.cat([embedded.unsqueeze(dim=1) for embedded in embeddeds], dim=1)

        h_pools = []
        for i, filter_height in enumerate(self.filter_heights):
            h_conv = F.relu(self.convs[i](embeddeds))
            h_pool = F.max_pool2d(h_conv, (self.max_sentence_len+1-filter_height, 1))
            h_pools.append( h_pool.squeeze(dim=3).squeeze(dim=2) )

        h_l1 = self.linear1(torch.cat(h_pools, dim=1))
        h_l1 = F.dropout(F.tanh(h_l1), p=0.5, training=training)
        h_l2 = self.linear2(h_l1)

        return F.softmax(h_l2, dim=1)

    def add_modules(self, name_prefix, modules):
        for i, module in enumerate(modules):
            self.add_module(name_prefix + str(i+1), module)

def make_XY(data, max_sentence_len):
    inputs, labels = [], []
    for d in data:
        # padding or cutting
        inputs.append(
                d["x"][:max_sentence_len] +
                [0] * (max_sentence_len - len(d["x"]))
                )
        labels.append(d["y"])
    inputs = torch.tensor(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    return inputs, labels

if __name__ == "__main__":
    num_words = 10
    num_classes = 4
    embedding_size = 20
    input_channel = 2
    output_channel = 5
    filter_heights = [2, 3]
    max_sentence_len = 4
    model = CNNClassifier(num_words, embedding_size, input_channel, output_channel, filter_heights, max_sentence_len, num_classes).to(device)

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
    inputs, labels = make_XY(data_train, max_sentence_len)

    for epoch in range(0, 100):
        optimizer.zero_grad()
        outputs = model(inputs, training=True)
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
    inputs, labels = make_XY(data_test, max_sentence_len)

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for predict, label in zip(predicted, labels):
            print("%d is predicted to %d" % (label, predict))
