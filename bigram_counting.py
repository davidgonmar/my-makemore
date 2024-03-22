from util import read_names
import torch
from typing import List

names = read_names()


class BigramByCounting:
    def __init__(self):
        self._fitted = False
        self._chars = None
        self._n_chars = None

    def stoi(self, char: str) -> int:
        return self._chars.index(char)

    def itos(self, idx: int) -> str:
        return self._chars[idx]

    def fit(self, names: List[str]):
        self._chars = list(sorted(set("".join(names))))
        self._chars.insert(0, ".")
        self._n_chars = len(self._chars)

        # occurences tensor. t[i, j] means 'ocurrences that char j occured after char i'
        # where char x means itos(x)
        t = torch.zeros(self._n_chars, self._n_chars, dtype=torch.float32)
        for name in names:
            name = "." + name + "."
            for char1, char2 in zip(name, name[1:]):
                idx1, idx2 = self.stoi(char1), self.stoi(char2)
                t[idx1, idx2] += 1

        # now, we need to convert t to a probability tensor so that
        # t[i, j] means 'what is the probability of char j occuring after char i
        # that is P(j|i)
        p = t / (t.sum(axis=1)).view(
            -1, 1
        )  # the view(-1, 1) is to make the division broadcast correctly

        self._fitted = True
        self._p = p
        self._t = t

    def predict(self, initial_char):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        prediction = "." + initial_char
        while prediction[-1] != ".":
            last_tok = self.stoi(prediction[-1])
            probabs = self._p[last_tok, :]
            pred = torch.multinomial(probabs, 1, True).item()
            prediction += self.itos(pred)
        prediction = prediction[1:-1]  # remove the dots
        return prediction


if __name__ == "__main__":
    model = BigramByCounting()
    model.fit(names)
    while True:
        inp = input("Enter a character: ")
        if not inp:
            break
        print(model.predict(inp))
