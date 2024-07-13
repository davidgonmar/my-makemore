from util import read_names
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple


torch.manual_seed(0)
names = read_names()

chars = list(sorted(set("".join(names))))

chars.insert(0, ".")  # Terminator (start and end) char
stoi = {s: idx for idx, s in enumerate(chars)}
itos = {idx: s for idx, s in enumerate(chars)}

n_chars = len(chars)


# shape[1] since we will transpose the matrix during the forward pass
def kaiming_init_tanh_fanin(shape):
    return torch.randn(shape) * (5 / 3) / (shape[1] ** 0.5)


class NeuralProbLM(nn.Module):
    def __init__(
        self, vocab_size: int, vocab_dimensionality: int, seq_len: int, hidden_size
    ):
        """
        Initializes the NeuralProbLM model as described in `A Neural Probabilistic Language Model` by Bengio et al (slightly modified)

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
        self.embed = nn.Parameter(
            torch.randn(vocab_size, vocab_dimensionality), requires_grad=True
        )
        self.H = nn.Parameter(
            kaiming_init_tanh_fanin((hidden_size, seq_len * vocab_dimensionality)),
            requires_grad=True,
        )
        self.U = nn.Parameter(
            kaiming_init_tanh_fanin((vocab_size, hidden_size)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.vocab_dimensionality = vocab_dimensionality
        self.hidden_size = hidden_size

        # learnable parameters for batch normalization
        self.bn_gain = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.bn_bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        # running averages for batch normalization during inference
        self.stat_change_rate = 0.01
        # technically, they are still part of the model, but not updated during backpropagation
        self.register_buffer("running_mean", torch.zeros(hidden_size))
        self.register_buffer("running_var", torch.ones(hidden_size))

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

        # First, we get embeddings of shape (batch_size, seq_len, vocab_dimensionality), where each row represents the embedding of the character,
        # so total number of columns is vocab_size
        # Then, we flatten the embeddings to (batch_size, seq_len * vocab_dimensionality)
        flat_embeddings = (one_hot_x @ self.embed).reshape(x.shape[0], -1)

        # Shape (batch_size, hidden_size)
        preacts = (
            flat_embeddings @ self.H.T
        )  # we dont use bias because we use batch normalization, and it gets 'absorbed' into the batch norm parameters.
        # The paper that introduced batch normalization (https://arxiv.org/abs/1502.03167) explains it.

        # batch normalization (with learnable parameters!) (dim 0 is the batch dimension), only during training
        if self.training:
            mean = preacts.mean(dim=0)
            var = preacts.var(dim=0)
            norm_preacts = (
                self.bn_gain * (preacts - mean) / (var + 1e-5).sqrt() + self.bn_bias
            )  # centers and scales to have mean 0 and std (and variance) close to 1
            # update running averages (for inference time), but we dont need to compute gradients for this
            with torch.no_grad():
                self.running_mean = (
                    self.stat_change_rate * mean
                    + (1 - self.stat_change_rate) * self.running_mean
                )
                self.running_var = (
                    self.stat_change_rate * var
                    + (1 - self.stat_change_rate) * self.running_var
                )

        else:
            # use running averages for batch normalization in inference time
            norm_preacts = (
                self.bn_gain
                * (preacts - self.running_mean)
                / (self.running_var + 1e-5).sqrt()
                + self.bn_bias
            )

        # non-linearity
        acts = torch.tanh(norm_preacts)

        # output computation
        out = acts @ self.U.T + self.b

        return out  # we dont compute softmax so these arent probabilities yet!


def build_dataset(words: List[str], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []
    # creates pairs of sequences of length seq_len and the next character
    for w in words:
        # for example, for 'hello', x will be
        """
        (['.', '.', '.'], 'h')
        (['.', '.', 'h'], 'e')
        (['.', 'h', 'e'], 'l')
        (['h', 'e', 'l'], 'l')
        (['e', 'l', 'l'], 'o')
        (['l', 'l', 'o'], '.')
        """
        context = [stoi["."]] * seq_len
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def regularization_loss(*args) -> torch.Tensor:
    return sum((x.pow(2).mean() for x in args))


def lr(epoch: int) -> float:
    # learning rate schedule
    return 0.1 / (1 + 0.1 * (epoch // 4000))


def train(model: NeuralProbLM, names: List[str]):
    seq_len = 3
    x, y = build_dataset(names, seq_len)
    batch_size = 32
    epochs = 300000
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100000, gamma=0.1)
    model.train()
    avg_loss = 0
    for epoch in range(epochs):
        optim.param_groups[0]["lr"] = lr(epoch)
        idx = torch.randint(0, x.shape[0], (batch_size,))
        x_batch = x[idx]
        y_batch = y[idx]

        out = model(x_batch)

        loss = F.cross_entropy(out, y_batch) + 0.001 * regularization_loss(
            model.embed, model.H, model.U
        )

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
    model = NeuralProbLM(n_chars, vocab_dimensionality=18, seq_len=3, hidden_size=200)
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
