================
Latent Diffusion Models for Generation of Neural Network Parameters
================

Research project examining the power of latent diffusion models for neural
network parameter generation from hyper-representations.

Description
===========

In this repository we are implementing a novel approach to "learning" neural network parameters.
For this we adapt the approach of latent diffusion models to model the learning process of
neural networks. We do this by
utilizing the LDM from `Rombach et al. 2022 <https://github.com/CompVis/stable-diffusion>`_
and use it together with the `ModelZooDataset <https://github.com/ModelZoos/ModelZooDataset>`_
which we transform with the help of `Kschuerholt et al.'s Hyperrepresentations <https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning>`_.


.. _pyscaffold-notes:

Development
===========
To start development start with

.. code-block:: console

    pip install -e ".[develop]"

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
