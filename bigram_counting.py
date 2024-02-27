from util import read_names
import torch
 
names = read_names()

chars = list(sorted(set(''.join(names))))

chars.insert(0, '.') # Terminator (start and end) char
stoi = {s:idx for idx, s in enumerate(chars)}
itos = {idx:s for idx, s in enumerate(chars)}

n_chars = len(chars)

# probability tensor. t[i, j] means 'ocurrences that char j occured after char i'
# where char x means itos(x)
t = torch.zeros(n_chars, n_chars, dtype=torch.float32)

for name in names:
    name = '.' + name + '.'
    for char1, char2 in zip(name, name[1:]):
        idx1, idx2 = stoi[char1], stoi[char2]
        t[idx1, idx2] += 1

# now, we need to convert t to a probability tensor so that
# t[i, j] means 'what is the probability of char j occuring after char i
# that is P(j|i)
eps = 1e-8
p = t / (t.sum(axis=1) + eps).view(-1, 1) # the view(-1, 1) is to make the division broadcast correctly

def predict(initial_char):
    prediction = '.' + initial_char
    while (prediction[-1] != '.'):
        last_tok = stoi[prediction[-1]]
        probabs = p[last_tok, :]
        pred = torch.multinomial(probabs, 1, True).item()
        prediction += itos[pred]
    return prediction


if __name__ == '__main__':
    while True:
        inp = input('Enter a character: ')
        if not inp: break
        print(predict(inp))