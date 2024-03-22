# My makemore

Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), this is a collection of different models that generate names given a dataset of names. It can be generalized to any dataset of strings.

## Models

- BigramByCounting: Simple model trained by counting bigrams in the dataset, and sampling from the distribution of bigrams.
- BigramNN: Same as BigramByCounting, but using a SGD trained neural network to predict the next character given the previous one.
- RNN: A simple RNN model that predicts the next character given the previous SEQ_LEN characters.
- LSTM: A simple LSTM model that predicts the next character given the previous SEQ_LEN characters.
- GRU: A simple GRU model that predicts the next character given the previous SEQ_LEN characters.

## Usage

```bash
python <model>.py
```

Will train the model and generate names, or ask the user for an initial sequence to generate names from.

## Installation

### Requirements

- Python 3.11
- PyTorch 2.1

```bash
pip install -r requirements.txt
```

You can use other kind of setup, like a virtual environment, or a conda environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
