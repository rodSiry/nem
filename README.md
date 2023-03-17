<h1>
  <a href="#"><img alt="banner" src="nem.gif" width="15%"/></a>
</h1>
*1st layer filters learnt continually by the NEM update rule on class-incremental MNIST sequence*

# nem

Implementation of NEM (Neurons for Emergent Memorization)

NEM is a neuron model whom inference and update rules are meta-trained to achieve efficient continual learning as an emergent property.

# installation

# use 

Run `nem.py` to launch the genetic search

Run `nem_test.py` to evaluate a trained update rule on various meta-test tasks (MNIST, SVHN...) and show learned filters

# reference 

The original paper of NEM is now outdated as it uses gradient-descent meta-optimization instead of black-box, genetic optimization, which is more stable 

Please refer (and cite) :

https://arxiv.org/abs/2111.02557
