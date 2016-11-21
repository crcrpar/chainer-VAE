# chainer-VAE
implementation of https://github.com/pfnet/chainer/tree/master/examples/vae using Trainer.

result/loss.png was compiled by using [Viz.js](https://mdaines.github.io/viz.js/).
If you use homebrew and wanna dump image file, `brew install graphviz`, then `dot -T <jpeg, png , etc> <computational_graph.dot> -o <output_image>`.

## beta-VAE
As you can see in http://openreview.net/forum?id=Sy2fzU9gl, you can impose stronger regularization on latent space when define model via `net.VAE(C=beta)`, where, beta > 1.


### Execute `train_vae.py`
#### Argument Parser
- resume: resume from snapshot. Basically, snapshots are `chainer-VAE/result/snapshot_epoch_{epoch}`
- interval: save images & snapshot intervals, default is `5`
