

.. _example_model_selection_grid_search_digits.py:


============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.grid_search.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.



**Python source code:** :download:`grid_search_digits.py <grid_search_digits.py>`

.. literalinclude:: grid_search_digits.py
    :lines: 18-
    