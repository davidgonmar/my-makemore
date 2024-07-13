import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer import Trainer
from data import get_names_dataloaders

torch.manual_seed(0)


class RNNLangModel(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, hidden_units: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_units = hidden_units
        self.l1 = nn.Linear(embed_dim + hidden_units, hidden_units)
        self.l2 = nn.Linear(hidden_units, vocab_size)

    def forward(self, input: torch.Tensor):
        # output is of size batch_size, vocab_size, representing probabilities for next word in vocab
        # input is a vector of shape batch_size, sequence_length
        batch_size, seq_len = input.size()

        # shape (batch_size, seq_len, embed_dim)
        tokens = self.embed(input)

        # shape (hidden_units)
        hiddens = [torch.zeros(batch_size, self.hidden_units)]
        for t in range(seq_len):
            # follows nomenclature in paper
            w = tokens[:, t, :]  # shape (batch_size, embed_dim)
            x = torch.cat(
                [hiddens[-1], w], dim=1
            )  # shape (batch_size, embed_dim + hidden_units)
            s = F.sigmoid(self.l1(x))  # shape (batch_size, hidden_units)
            hiddens.append(s)

        # output is not probabilities, but logits (the paper uses softmax, but I decided to leave it out of the model)
        return self.l2(
            torch.stack(hiddens[1:], dim=1)
        )  # batch_size, seq_len, vocab_size

    def generate(self, input: torch.Tensor):
        # shape (batch_size, seq_len)
        assert (
            input.dim() == 2 and input.size(1) == 1 and input.size(0) == 1
        ), "input must be of shape (1, 1)"
        tok = self.embed(input)
        w = tok[:, 0, :]  # shape (1, embed_dim)
        # shape (hidden_units)
        lasthidden = torch.zeros(1, self.hidden_units)
        out = []
        t = 0
        while True:
            x = torch.cat([lasthidden, w], dim=1)
            s = F.sigmoid(self.l1(x))
            lasthidden = s
            res = self.l2(s)
            resprobs = F.softmax(res, dim=1)
            sampled = torch.multinomial(resprobs, 1).reshape(1)
            out.append(sampled)
            w = self.embed(sampled)
            if out[-1].item() == 0:
                break
            t += 1
        return torch.stack(out, dim=0).reshape(-1)


if __name__ == "__main__":
    train_dataloader, test_dataloader = get_names_dataloaders(seq_len=3, batch_size=128)
    model = RNNLangModel(10, train_dataloader.dataset.n_chars, 100)
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
        res = model.generate(
            torch.tensor(train_dataloader.dataset.stoi["."]).reshape(1, 1)
        )
        print(
            "".join([train_dataloader.dataset.itos[i.item()] for i in res]).replace(
                ".", ""
            )
        )
