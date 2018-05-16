#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_words, embedding_size, output_channel, filter_height, max_sentence_len, num_classes):
        super(SimpleCNNClassifier, self).__init__()

        input_size = num_words + 1 # including pad
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)

        input_channel = 1
        self.conv = nn.Conv2d(input_channel, output_channel, (filter_height, embedding_size), padding=0)
        self.pool_kernel_size = (max_sentence_len+1-filter_height, 1)

        self.linear = nn.Linear(output_channel, num_classes)

    def forward(self, input):
        embedded = self.embedding(input)

        h_conv = F.relu(self.conv(embedded.unsqueeze(dim=1)))
        h_pool = F.max_pool2d(h_conv, self.pool_kernel_size)

        output = self.linear(h_pool.squeeze(dim=3).squeeze(dim=2))

        return F.softmax(output, dim=1)

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
    output_channel = 5
    filter_height = 2
    max_sentence_len = 4
    model = SimpleCNNClassifier(num_words, embedding_size, output_channel, filter_height, max_sentence_len, num_classes).to(device)

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
    inputs, labels = make_XY(data_test, max_sentence_len)

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for predict, label in zip(predicted, labels):
            print("%d is predicted to %d" % (label, predict))
