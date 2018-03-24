# NeuralTQE
An extension of the MT Quality Estimation model from [https://github.com/hittle2015/NeuralTQE]

# Method
The basic idea is to use cross-attention over the vectors produced for the source sentences and their MT versions. The paper (Yuan, Sharoff, 2018) will be available soon.

# Training and Testing Data Format
Training and Testing data should be in the format as:
```
source_sentence \t target_sentence \t score
```

Note that all sentences should be tokenised.

# Run the Code
The basic setup is:
```
python mtmain.py -m ModelFile -t de_en.train -v de_en.dev -1 de-300.vec -2 en-300.vec -o Predictions
```
For more info about the training parameters, run the standard:
`python mtmain.py -h`

# Requirements
1. Numpy
2. PyTorch

It should work in either python2 or python3.