"""Based on the paper Recurrent Neural Network Based Language Model - https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf"""
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
        prev_hidden = torch.zeros(batch_size, self.hidden_units)

        output = None
        for t in range(seq_len):
            # follows nomenclature in paper
            w = tokens[:, t, :]  # shape (batch_size, embed_dim)
            x = torch.cat(
                [prev_hidden, w], dim=1
            )  # shape (batch_size, embed_dim + hidden_units)
            s = F.sigmoid(self.l1(x))  # shape (batch_size, hidden_units)
            prev_hidden = s
            # only predict for last word!!
            if t is seq_len - 1:
                output = self.l2(s)  # shape (batch_size, vocab_size)

        # output is not probabilities, but logits (the paper uses softmax, but I decided to leave it out of the model)
        return output  # batch_size, vocab_size


@torch.no_grad()
def predict(model, seq_len, stoi, itos):
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


if __name__ == "__main__":
    train_dataloader, test_dataloader = get_names_dataloaders(seq_len=3, batch_size=128)
    model = RNNLangModel(10, train_dataloader.dataset.n_chars, 100)

    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters()),
        nn.CrossEntropyLoss(),
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
