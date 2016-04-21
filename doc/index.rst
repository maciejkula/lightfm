.. LightFM documentation master file, created by
   sphinx-quickstart on Thu Apr 21 12:26:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LightFM's documentation!
===================================

LightFM, is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

The details of the approach are described in the LightFM paper, available on `arXiv <http://arxiv.org/abs/1507.08439>`_.

Installation
============

Install from pypi using pip: ``pip install lightfm``. Everything should work out-of-the box on Linux and Windows using MSVC.

Note for OSX users: due to its use of OpenMP, LightFM does not compile under Clang. To install it, you will need a reasonably recent version of ``gcc`` (from Homebrew for instance). This should be picked up by ``setup.py``; if it is not, please open an issue.

Building with the default Python distribution included in OSX is also not supported; please try the version from Homebrew or Anaconda.

If you are having problems, you should try using lightfm with Docker (see the next section).

Using with Docker
-----------------

On many systems it may be more convenient to try LightFM out in a Docker container. This repository provides a small Dockerfile sufficient to run LightFM and its examples. To run it:

1. `Install Docker <https://docs.docker.com/compose/install/>`_ and start the docker deamon/virtual machine.
2. Clone this repository and navigate to it: ``git clone git@github.com:lyst/lightfm.git && cd lightfm``.
3. Run ``docker-compose build lightfm`` to build the container.

The container should now be ready for use. You can then:

1. Run tests by running ``docker-compose run lightfm py.test -x tests/``
2. Run the movielens example by running ``docker-compose run lightfm jupyter notebook examples/movielens/example.ipynb --ip=0.0.0.0``. The notebook will be accessible at port 8888 of your container's IP address.

Usage
=====

Model fitting is very straightforward using the main :doc:`LightFM class <lightfm>`.

Create a model instance with the desired latent dimensionality::

    from lightfm import LightFM

    model = LightFM(no_components=30)

Assuming ``train`` is a (no_users, no_items) sparse matrix (with 1s denoting positive, and -1s negative interactions), you can fit a traditional matrix factorization model by calling::

    model.fit(train, epochs=20)

This will train a traditional MF model, as no user or item features have been supplied.

To get predictions, call ``model.predict``::

    predictions = model.predict(test_user_ids, test_item_ids)


User and item features can be incorporated into training by passing them into the ``fit`` method. Assuming ``user_features`` is a (no_users, no_user_features) sparse matrix (and similarly for ``item_features``), you can call::

    model.fit(train,
              user_features=user_features,
              item_features=item_features,
              epochs=20)
    predictions = model.predict(test_user_ids,
                                test_item_ids,
                                user_features=user_features,
                                item_features=item_features)

to train the model and obtain predictions.

Both training and prediction can employ multiple cores for speed::

    model.fit(train, epochs=20, num_threads=4)
    predictions = model.predict(test_user_ids, test_item_ids, num_threads=4)

This implementation uses asynchronous stochastic gradient descent [6] for training. This can lead to lower accuracy when the interaction matrix (or the feature matrices) are very dense and a large number of threads is used. In practice, however, training on a sparse dataset with 20 threads does not lead to a measurable loss of accuracy.

In an implicit feedback setting, the BPR, WARP, or k-OS WARP loss functions can be used. If ``train`` is a sparse matrix with positive entries representing positive interactions, the model can be trained as follows::

    model = LightFM(no_components=30, loss='warp')
    model.fit(train, epochs=20)


Examples
========

Check the ``examples`` directory for more examples.

The `Movielens example <https://github.com/lyst/lightfm/blob/master/examples/movielens/example.ipynb>`_ shows how to use LightFM on the Movielens dataset, both with and without using movie metadata. `Another example <https://github.com/lyst/lightfm/blob/master/examples/movielens/learning_schedules.ipynb>`_ compares the performance of the adagrad and adadelta learning schedules.

The `Cross Validated example <https://github.com/lyst/lightfm/blob/master/examples/crossvalidated/example.ipynb>`_ shows how to use LightFM on a dataset from `stats.stackexchange.com <http://stats.stackexchange.com>`_ with both item and user features.

The `Kaggle coupon purchase prediction <https://github.com/tdeboissiere/Kaggle/blob/master/Ponpare/ponpare_lightfm.ipynb>`_ example applies LightFM to predicting coupon purchases.

Development
===========

Pull requests are welcome. To install for development:

1. Clone the repository: ``git clone git@github.com:lyst/lightfm.git``
2. Install it for development using pip: ``cd lightfm && pip install -e .``
3. You can run tests by running ``python setupy.py test``.

When making changes to the ``.pyx`` extension files, you'll need to run ``python setup.py cythonize`` in order to produce the extension ``.c`` files before running ``pip install -e .``.


Contents
========

.. toctree::
   :maxdepth: 2

   LightFM model <lightfm>
   Model evaluation <lightfm.evaluation>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

