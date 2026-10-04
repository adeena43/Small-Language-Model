
| property (per layer)             | RNN            | Self-attention |
|----------------------------------|----------------|----------------|
| sequential operations            | O(T)           | O(1)           |
| compute per layer                | O(T * d^2)     | O(T^2 * d)     |
| memory for pairwise scores       | none           | O(T^2) per head|
| max path length between 2 tokens | O(T)           | O(1)           |
