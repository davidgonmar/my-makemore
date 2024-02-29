from util import read_names
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple

names = read_names()

chars = list(sorted(set("".join(names))))

chars.insert(0, ".")  # Terminator (start and end) char
stoi = {s: idx for idx, s in enumerate(chars)}
itos = {idx: s for idx, s in enumerate(chars)}

n_chars = len(chars)


class NeuralProbLM(nn.Module):
    def __init__(
        self, vocab_size: int, vocab_dimensionality: int, seq_len: int, hidden_size
    ):
        """
        Initializes the NeuralProbLM model.

        Args:
        vocab_size: An integer representing the number of characters in the vocabulary.
        vocab_dimensionality: An integer representing the dimensionality of the character embeddings ('how many features to represent each character')
        seq_len: An integer representing the length of the sequence to predict (for example, if seq_len is 3, then the model will predict the next character given the previous 3 characters)
        hidden_size: An integer representing the number of hidden units in the model.
        """

        super().__init__()

        # as the paper A mapping C from any element i of V to a real vector C(i) ∈ Rm. It represents the distributed
        # feature vectors associated with each word in the vocabulary. In practice, C is represented by
        # a |V| ×m matrix of free parameters
        self.embed = nn.Parameter(torch.randn(vocab_size, vocab_dimensionality))
        self.H = nn.Parameter(torch.randn(hidden_size, seq_len * vocab_dimensionality))
        self.d = nn.Parameter(torch.zeros(hidden_size))
        self.U = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.b = nn.Parameter(torch.randn(vocab_size))
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.vocab_dimensionality = vocab_dimensionality
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor):
        """
        Args:
        x: A tensor with shape (batch_size, seq_len), representing a batch of sequences of characters.
        They must lie in the range [0, vocab_size), and each row in x represents a sequence of characters.

        Returns:
        A tensor with shape (batch_size, vocab_size). Each row represents the raw output of the model for each sequence in the batch.
        """
        # Shape (batch_size, seq_len, vocab_size), where each row represents a character, one hot encoded
        # so the '1' is at the position of the character (and therefore, the rest are 0s).
        # This serves as a way to gather the embeddings in a batched way
        one_hot_x = F.one_hot(x, self.vocab_size).float()

        # Shape (batch_size, seq_len, vocab_dimensionality), where each row represents the embedding of the character,
        # so total number of columns is vocab_size
        embeddings = one_hot_x @ self.embed

        flat = torch.reshape(
            embeddings, (x.shape[0], -1)
        )  # flatten the embeddings to a shape of (batch_size, seq_len * vocab_dimensionality)

        a = torch.tanh(flat @ self.H.T + self.d)  # shape (batch_size, hidden_size)
        a = a @ self.U.T + self.b  # shape (batch_size, vocab_size)

        return a  # we dont compute softmax so these arent probabilities yet!


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


def train(model: NeuralProbLM, names: List[str]):
    seq_len = 3
    x, y = build_dataset(names, seq_len)
    batch_size = 32
    epochs = 30000
    optim = torch.optim.SGD(model.parameters(), lr=0.2)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10000, gamma=0.1)
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


def predict(model, seq_len):
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
    return "".join(itos[i] for i in result)


def test(model, inputs, targets):
    out = model(inputs)
    loss = F.cross_entropy(out, targets)
    print(f"Test loss: {loss.item()}")


if __name__ == "__main__":
    model = NeuralProbLM(n_chars, vocab_dimensionality=2, seq_len=3, hidden_size=200)
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
