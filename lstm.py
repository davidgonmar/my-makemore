import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_names_dataloaders
from trainer import Trainer
from util import read_names
from typing import List, Tuple

torch.manual_seed(0)
names = read_names()

chars = list(sorted(set("".join(names))))

chars.insert(0, ".")  # Terminator (start and end) char
stoi = {s: idx for idx, s in enumerate(chars)}
itos = {idx: s for idx, s in enumerate(chars)}

n_chars = len(chars)


class LSTMLangModel(nn.Module):
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
        prev_c = torch.zeros(batch_size, self.hidden_units)
        hiddens = [torch.zeros(batch_size, self.hidden_units)]

        for t in range(seq_len):
            xt = tokens[:, t, :]
            catted = torch.cat([xt, hiddens[-1]], dim=1)
            forget_gate_out = self.forget_gate(catted)
            prev_c = prev_c * forget_gate_out
            input_gate_out = self.input_gate(catted)
            candidate_mem = self.candidate_mem_layer(catted)
            prev_c = prev_c + candidate_mem * input_gate_out
            output_gate_out = self.output_gate(catted)
            hiddens.append(prev_c * output_gate_out)

        # output is not probabilities, but logits (the paper uses softmax, but I decided to leave it out of the model)
        return self.to_out(
            torch.stack(hiddens[1:], dim=1)
        )  # batch_size, seq_len, vocab_size


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


def train(model: LSTMLangModel, names: List[str]):
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
def predict(model, seq_len, stoi, itos):
    model.eval()
    # start with '.' as the first character
    input = torch.tensor([[stoi["."]]])
    out = [itos[input.item()]]
    for _ in range(seq_len):
        out_logits = model(input)
        # get the last token
        input = torch.argmax(out_logits[:, -1, :], dim=1).unsqueeze(1)
        out.append(itos[input.item()])

    return "".join(out).replace(".", "")


if __name__ == "__main__":
    train_dataloader, test_dataloader = get_names_dataloaders(seq_len=3, batch_size=128)
    model = LSTMLangModel(10, train_dataloader.dataset.n_chars, 100)
    lfn = nn.CrossEntropyLoss()

    def loss_fn(out, target):
        # out of shape (batch_size * seq_len, vocab_size)
        # target of shape (batch_size * seq_len)
        return lfn(out.view(-1, out.size(2)), target.view(-1))

    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters()),
        loss_fn,
        train_dataloader,
        test_dataloader,
    )
    trainer.train(epochs=100)
    trainer.validate()
    # ask user for input
    for _ in range(20):
        print(
            predict(
                model, 3, train_dataloader.dataset.stoi, train_dataloader.dataset.itos
            )
        )
