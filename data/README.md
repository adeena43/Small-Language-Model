# Data

**Dataset:** Tiny Shakespeare — ~1.1 MB of text (~40,000 lines) from a variety of Shakespeare plays.
Source: Hugging Face dataset card <https://huggingface.co/datasets/karpathy/tiny_shakespeare>
(original text from Andrej Karpathy's `char-rnn` repository).

The file `data/input.txt` is **downloaded automatically** by `src/dataset.py` the first time you train, so it is not
committed to git. To fetch it manually:

```bash
curl -L -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Split: first 90 % of the characters = train, last 10 % = validation (contiguous, no shuffling across the boundary).
Tokenizer: character level, 65-character vocabulary, no special tokens (see `src/tokenizer.py`).
