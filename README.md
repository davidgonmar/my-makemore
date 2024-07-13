# My makemore

Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), this is a collection of different models that generate names given a dataset of names. It can be generalized to any dataset of strings.

## Models

- BigramByCounting: Simple model trained by counting bigrams in the dataset, and sampling from the distribution of bigrams.
- BigramNN: Same as BigramByCounting, but using a SGD trained neural network to predict the next character given the previous one.
- MLP: A model, following (slightly modified) [A Neural Probabilistic Language Model, Bengio et al.](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
- RNN, LSTM, GRU: Recurrent neural network models, following the well-known architectures.

## Usage

To run any of the scripts, just run:

```bash
python <model>.py
```

Will train the model and generate names, or ask the user for an initial sequence to generate names from.

## Installation

### Requirements

Python 3.11.5 was used to develop this project.
The requirements can be installed with:

```bash
pip install -r requirements.txt
```

You can use other kind of setup, like a virtual environment, or a conda environment.

## References

- [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)
- [Andrej Karpathy's YouTube series](https://www.youtube.com/@AndrejKarpathy)
- [A Neural Probabilistic Language Model, Bengio et al.](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
- [Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks](https://arxiv.org/abs/1701.05923)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
