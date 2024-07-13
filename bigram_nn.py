from torch import nn
import torch
from util import read_names
import torch.nn.functional as F


names = read_names()

chars = list(sorted(set("".join(names))))

chars.insert(0, ".")  # Ttrminator (start and end) char
stoi = {s: idx for idx, s in enumerate(chars)}
itos = {idx: s for idx, s in enumerate(chars)}

n_chars = len(chars)


class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initializes the Bigram model.

        Args:
        vocab_size: An integer representing the number of characters in the vocabulary.
        """
        super().__init__()
        # simply a lookup table such that lt[i, j] means
        # what is the probability of the character j appearing after character i
        self.lt = nn.Parameter(torch.randn(vocab_size, vocab_size), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: A one-hot encoded tensor with shape (sequence_length, vocab_size), representing
            a sequence of characters. Each row in x is a one-hot encoded vector representing
            a character in the sequence.

        Returns: A tensor of log probabilities with shape (sequence_length, vocab_size),
            where each row corresponds to an input character (of x), and each column
            represents the log probability of each vocabulary character following
            the input character.
        """

        # Multiply the input one-hot encoded tensor by the lookup table to retrieve the
        # probabilities for each character encoded in the one-hot tensor
        return x @ self.lt


lr_dict = {
    **{i: 100 for i in range(50)},
    **{i: 80 for i in range(50, 150)},
    **{i: 70 for i in range(150, 250)},
    **{i: 60 for i in range(250, 350)},
    **{i: 50 for i in range(350, 500)},
}


def train(epochs: int):
    model = Bigram(n_chars)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    inputs = []
    targets = []

    for name in names:
        name = "." + name + "."
        for char1, char2 in zip(name, name[1:]):
            inputs.append(stoi[char1])
            targets.append(stoi[char2])

    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    for epoch in range(epochs):
        # set the learning rate
        optimizer.param_groups[0]["lr"] = lr_dict.get(epoch, 1)

        oh_inputs = F.one_hot(inputs, num_classes=n_chars).float()

        outs = model.forward(oh_inputs)

        # negative log likelihood (model doesnt output softmaxed)
        cse = nn.CrossEntropyLoss()

        loss = cse(outs, targets) + 0.001 * model.lt.pow(2).mean()  # L2 regularization

        loss.backward()

        optimizer.step()

        model.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs} done, loss: {loss.item()}")

    return model


def predict(initial_char, model):
    prediction = "." + initial_char

    while prediction[-1] != ".":
        last_tok = stoi[prediction[-1]]
        out = model.forward(
            F.one_hot(torch.tensor([last_tok]), num_classes=n_chars).float()
        )
        probabs = F.softmax(out, dim=1)
        pred = torch.multinomial(probabs, 1, True).item()
        prediction += itos[pred]

    return prediction.replace(".", "")


if __name__ == "__main__":
    trained = train(500)

    # predict
    while True:
        inp = input("Enter a character: ")
        if not inp:
            break
        print(predict(inp, trained))
