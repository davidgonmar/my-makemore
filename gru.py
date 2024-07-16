import torch
import torch.nn as nn
from data import get_names_dataloaders
from trainer import Trainer

torch.manual_seed(0)


class GRULangModel(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, hidden_units: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_units = hidden_units
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_units + embed_dim, hidden_units), nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_units + embed_dim, hidden_units), nn.Sigmoid()
        )
        self.candidate_state = nn.Sequential(
            nn.Linear(hidden_units + embed_dim, hidden_units), nn.Tanh()
        )
        self.to_out = nn.Linear(hidden_units, vocab_size)

    def forward(self, input: torch.Tensor):
        # output is of size batch_size, vocab_size, representing probabilities for next word in vocab
        # input is a vector of shape batch_size, sequence_length
        batch_size, seq_len = input.size()

        # shape (batch_size, seq_len, embed_dim)
        tokens = self.embed(input)
        hiddens = [torch.zeros(batch_size, self.hidden_units)]

        for t in range(seq_len):
            xt = tokens[:, t, :]  # shape (batch_size, embed_dim)
            catted = torch.cat([hiddens[-1], xt], dim=1)
            reset_gate_out = self.reset_gate(catted)
            reset_gate_out_times_prev_hidden = (
                reset_gate_out * hiddens[-1]
            )  # (batch_size, embed_dim)
            candidate_hidden_state = self.candidate_state(
                torch.cat([reset_gate_out_times_prev_hidden, xt], dim=1)
            )
            update_gate_out = self.update_gate(catted)
            h = hiddens[-1] * update_gate_out
            h = (1 - update_gate_out) * candidate_hidden_state + h
            hiddens.append(h)

        # output is not probabilities, but logits (the paper uses softmax, but I decided to leave it out of the model)
        return self.to_out(
            torch.stack(hiddens[1:], dim=1)
        )  # batch_size, seq_len, vocab_size


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
    model = GRULangModel(10, train_dataloader.dataset.n_chars, 100)
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
