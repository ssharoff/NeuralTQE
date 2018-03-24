# NeuralTQE
An extension of MT Quality Estimation model from [hittle2015/NeuralTQE]

# Method
The basic idea is to use cross-attention over the vectors produced for the source sentences and their MT versions.

# Training and Testing Data Format
Training and Testing data should be in the format as :
source_sentence \t target_sentence \t \ score
Note that all sentences should be segmented.

# Run the Code
The basic setup is:
```
python mtmain.py -m ModelFile -t de_en.train -v de_en.dev -1 de-300.vec -2 en-300.vec -o Predictions
```
For more info about the training parameters, run the standard:
`python mtmain.py -h`
