# AI that write verses 
***\*\*This repository is under construction!\*\****

## What is this?
This is just a fun project trying to create AI models that can "compose" verses.

Three models have been experimented:

  1. A **regular RNN** with GRU cells
  2. A **stateful RNN** (also with GRU cells)
  3. A **WaveNet-style model**

Using the given `train.py` you can easily create, customize and train these models. 

It's like a fun art to train these models: we want them to overfit the training set in order to get the author's "style" of writing, but we don't want it to be too overfitting in order to get some fresh novel text. Oh, is it the Bias-variance tradeoff in machine learning?

## Brief description of the training data
The training text is from ***Truyện Kiều*** (***The Tale of Kiều***)--a poem of 3,254 verses written in six–eigh meter. 

> "The Tale of Kiều is an epic poem in Vietnamese written by Nguyễn Du (1765–1820), considered the most famous poem and a classic in Vietnamese literature." (Wikipedia)

You can find the training data in `datasets` folder and trained models in `models` folder.

