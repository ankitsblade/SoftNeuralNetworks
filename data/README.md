# Data

`torchvision` will download CIFAR-10 under this directory by default.

Place CIFAR-10H soft targets at:

```text
data/cifar10h/cifar10h-probs.npy
```

The target file must contain a `10000 x 10` array aligned with the CIFAR-10 test set. Rows may be probabilities or annotator counts; the loader normalizes each row and checks that it sums to one.
