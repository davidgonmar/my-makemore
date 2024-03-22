"""Based on the paper Recurrent Neural Network Based Language Model - https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import read_names
from typing import List, Tuple

torch.manual_seed(0)
names = read_names()

chars = list(sorted(set("".join(names))))

chars.insert(0, ".")  # Terminator (start and end) char
stoi = {s: idx for idx, s in enumerate(chars)}
itos = {idx: s for idx, s in enumerate(chars)}

n_chars = len(chars)


class RNNLangModel(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, hidden_units: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_units = hidden_units
        self.candidate_mem_layer = nn.Sequential(
            nn.Linear(embed_dim + hidden_units, hidden_units), nn.Tanh()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(embed_dim + hidden_units, hidden_units), nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            nn.Linear(embed_dim + hidden_units, hidden_units), nn.Sigmoid()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(embed_dim + hidden_units, hidden_units), nn.Sigmoid()
        )

        self.to_out = nn.Linear(hidden_units, vocab_size)

    def forward(self, input: torch.Tensor):
        # output is of size batch_size, vocab_size, representing probabilities for next word in vocab
        # input is a vector of shape batch_size, sequence_length
        batch_size, seq_len = input.size()

        # shape (batch_size, seq_len, embed_dim)
        tokens = self.embed(input)

        # shape (hidden_units)
        prev_h = torch.zeros(batch_size, self.hidden_units)
        prev_c = torch.zeros(batch_size, self.hidden_units)

        output = None
        for t in range(seq_len):
            xt = tokens[:, t, :]
            catted = torch.cat([xt, prev_h], dim=1)
            forget_gate_out = self.forget_gate(catted)
            prev_c = prev_c * forget_gate_out
            input_gate_out = self.input_gate(catted)
            candidate_mem = self.candidate_mem_layer(catted)
            prev_c = prev_c + candidate_mem * input_gate_out
            output_gate_out = self.output_gate(catted)
            prev_h = prev_c * output_gate_out

            # only predict for last word!!
            if t is seq_len - 1:
                output = self.to_out(prev_h)  # shape (batch_size, vocab_size)

        # output is not probabilities, but logits (the paper uses softmax, but I decided to leave it out of the model)
        return output  # batch_size, vocab_size


def build_dataset(words: List[str], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []
    for w in words:
        context = [stoi["."]] * seq_len
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def train(model: RNNLangModel, names: List[str]):
    seq_len = 3
    x, y = build_dataset(names, seq_len)
    batch_size = 32
    epochs = 300000
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100000, gamma=0.2)
    model.train()
    avg_loss = 0
    for epoch in range(epochs):
        idx = torch.randint(0, x.shape[0], (batch_size,))
        x_batch = x[idx]
        y_batch = y[idx]

        out = model(x_batch)

        loss = F.cross_entropy(out, y_batch)

        optim.zero_grad()

        loss.backward()

        optim.step()
        sched.step()

        avg_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, loss: {avg_loss / 10}")
            avg_loss = 0


@torch.no_grad()
def predict(model, seq_len):
    model.eval()
    # init all with '.'
    context = [stoi["."]] * seq_len
    result = []
    while True:
        x = torch.tensor([context])
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred = torch.multinomial(probs, 1).item()
        if pred == 0:
            break
        context = context[1:] + [pred]
        result.append(pred)
    model.train()
    return "".join(itos[i] for i in result)


@torch.no_grad()
def test(model, inputs, targets):
    model.eval()
    out = model(inputs)
    loss = F.cross_entropy(out, targets)
    print(f"Test loss: {loss.item()}")
    model.train()


if __name__ == "__main__":
    model = RNNLangModel(10, n_chars, 100)
    # shuffle names
    import random

    random.shuffle(names)
    names_train = names[: int(len(names) * 0.9)]
    random.shuffle(names_train)
    names_test = names[int(len(names) * 0.9) :]
    train(model, names_train)
    test(model, *build_dataset(names_test, 3))

    # ask user for input
    for _ in range(20):
        print(predict(model, 3))
