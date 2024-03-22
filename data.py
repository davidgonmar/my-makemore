
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple
import random
torch.manual_seed(0)


NAMES_PATH = os.path.join(".", "names.txt")

def read_names() -> List[str]:
    with open(NAMES_PATH, "r") as f:
        names = f.read().splitlines()
    return names

class NamesDataset(Dataset):

    def __init__(self, names: str, seq_len: int = 3):
        """
        Args:
            path: path to the file containing the names
            seq_len: length of the sequence to consider
        """
        self.names = names
        self.chars = self._compute_chars(self.names)
        self.stoi = {s: idx for idx, s in enumerate(self.chars)}
        self.itos = {idx: s for idx, s in enumerate(self.chars)}
        self.n_chars = len(self.chars)
        self.X, self.Y = self._build_dataset(self.names, seq_len)


    def _compute_chars(self, names: List[str]) -> List[str]:
        x = list(sorted(set("".join(names))))
        x.insert(0, ".")
        return x
    
    def _build_dataset(self, words: List[str], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []
        for w in words:
            context = [self.stoi["."]] * seq_len
            for ch in w + ".":
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


names = read_names()

def get_names_dataloaders(seq_len: int = 3, split: float = 0.8, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    random.shuffle(names)
    names_train = names[:int(len(names) * split)]
    names_test = names[int(len(names) * split):]
    train_dataset = NamesDataset(names_train, seq_len)
    test_dataset = NamesDataset(names_test, seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader