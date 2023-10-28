# PAML 2023 Deep RL Intro
Introduction to Deep RL for PAML Group Seminars 2023.

Introduces how to quickly get started with Deep RL using PyTorch and Gymnasium.
The examples here implement basic Deep Q-Networks (DQN) for tasks with
continuous observation space and finite action space.

## Dependencies

The dependencies for this code is primarily PyTorch, Gymnasium and TensorBoard.
They can conveniently be installed with conda:

```sh
conda env create -f condaenv.yaml
```

## Running the Code

To run a training session on an environment like LunarLander (run `./train -h`
to see available environments), run

```
./train.py lunarlander
```

This will take roughly 5 minutes on a modern laptop. When it is done you
should see it output a file `lunarlander.pkl`. To evaluate your resulting
policy, run

```
./eval.py lunarlander.pkl
```

If you want to see more statistics from your training process, run

```
tensorboard --logdir=_tensorboard
```

and open up TensorBoard in your browser using the link that was outputted from
the `tensorboard` command above.

## Running the CNN Code

Training the CNN models in `train-cnn.py` requires quite a lot of computational
resources. It is recommended to have a computer that has at least 64 GiB of RAM
memory and a CUDA compatible GPU. Install the `condaenv-cuda.yaml` environment
on your system and manually install the commented pip-commands in the conda
file. Once trained, the models should be able to be evaluated on a modern
laptop.
