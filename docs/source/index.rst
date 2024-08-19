.. CoxKAN documentation master file, created by
   sphinx-quickstart on Tue Jun 25 12:20:20 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/coxkan_logo.png

-----------------------------------------------------

=======================
CoxKAN Documentation
=======================

CoxKAN is a Python package for performing survival analysis using Kolmogorov-Arnold Networks.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro.rst
   coxkan
   coxkan.datasets

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Installation
============

To install CoxKAN, use pip:

::

    pip install coxkan

Usage
=====

Here is a simple example of how to use CoxKAN:

.. code-block:: python

    from coxkan import CoxKAN
    from coxkan.datasets import metabric
    from sklearn.model_selection import train_test_split

    df = metabric.load()
    df_train, df_test = train_test_split(df, test_size=0.2)
    dataset_name, duration_col, event_col, covariates = dataset.metadata()

    ckan = CoxKAN(width=[len(covariates),2,1], grid=5, k=3)

    log = ckan.train(
        df_train, 
        df_test, 
        duration_col='duration', 
        event_col='event',
        opt='Adam',
        lr=0.01,
        steps=100)

For more information, visit the `CoxKAN GitHub repository <https://github.com/knottwill/CoxKAN>`_.

Contributing
============

Contributions welcome! Please submit a pull request or open an issue on GitHub.
