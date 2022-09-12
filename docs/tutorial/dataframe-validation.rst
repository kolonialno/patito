Using Patito for DataFrame Validation
=====================================

Have you ever found yourself relying on some column of an external data source being non-nullable only to find out `much` later that the assumption proved to be false?
What about discovering that a production model has had a huge performance regression because a new category was introduced to a categorical column, and that the training pipeline neglected to account for this new reality?

You might not have encountered any of these `exact` scenarios, but perhaps similar ones.
They illustrate the necessity of validating your assumptions.
It is much better to explicitly specify all the constraints of your data and let your program fail loud and clear when they do `not` hold, compared to letting errors go undetected and wreak havoc.

The `polars <https://github.com/pola-rs/polars>`_ dataframe library has been making the rounds lately among data scientists at Oda.
It can be considered as a total replacement of pandas, initially tempting you with its advertised `top-notch performance <https://www.pola.rs/benchmarks.html>`_, but then sealing the deal with its intuitive and expressive API.
The exact virtues of polars is a topic for another article, but suffice it to say that it is highly recommended to try it out, it has some great `introductory documentation <https://pola-rs.github.io/polars-book/user-guide/>`_.

At its core, this is the problem Patito tries to solve, it offers a `declarative` way to specify the constraints of your data in the form of :ref:`models <Model>`.
If you persistently use these models to validate the data sources wherever they enter the data pipeline, you will turn your `data assumptions` into `data assertions`.
In turn, your models become a trustworthy centralized catalog of all the core facts about your data, facts you can safely rely upon during development.

Enough chit chat, let's get into some technical details!
Let's say that your project keeps track of products, and that these products have three core properties:

1. A unique numeric identifier...
2. a name...
3. and an ideal temperature zone, one of either ``"dry"``, ``"cold"``, or ``"frozen"``.

In tabular form the data might look something like this.

.. _product_table:

.. list-table:: Table 1: Products
    :widths: 33 33 33
    :header-rows: 1
    :align: center

    * - ``product_id``
      - ``name``
      - ``temperature_zone``
    * - 1
      - Apple
      - dry
    * - 2
      - Milk
      - cold
    * - 3
      - Ice cubes
      - frozen
    * - ...
      - ...
      - ...

We now start to model the restrictions we want to put upon our data.
In Patito this is done by defining a class which inherits from ``patito.Model``, a class which has one `field annotation` for each column in the data.
These models should preferably be defined in a centralized place, conventionally ``<YOUR_PROJECT_NAME>/models.py``, where you can easily find and refer to them.

.. code-block:: python
   :caption: project/models.py

    from typing import Literal

    import patito as pt


    class Product(pt.Model):
        product_id: int
        name: str
        temperature_zone: Literal["dry", "cold", "frozen"]


Here we have used ``typing.Literal`` from `the standard library <https://docs.python.org/3/library/typing.html#typing.Literal>`_ in order to specify that ``temperature_zone`` is not only a ``str``, but `specifically` one of the literal values ``"dry"``, ``"cold"``, or ``"frozen"``.
You can now use this class to represent a `single specific instance` of a product:

.. code-block:: python

    >>> Product(product_id=1, name="Apple", temperature_zone="dry")
    Product(product_id=1, name='Apple', temperature_zone='dry')


The class also automatically offers input data validation, for instance if you provide an invalid value for ``temperature_zone``.

.. code-block:: python

    >>> Product(product_id=64, name="Pizza", temperature_zone="oven")
    ValidationError: 1 validation error for Product
    temperature_zone
      unexpected value; permitted: 'dry', 'cold', 'frozen' (type=value_error.const; given=oven; permitted=('dry', 'cold', 'frozen'))

A discerning reader might notice that this looks suspiciously like `pydantic's <https://github.com/pydantic/pydantic>`_ data models, and that is in fact because it is!
Patito's model class is built upon pydantic's ``pydantic.BaseClass`` and therefore offers `all of pydantic's functionality <https://pydantic-docs.helpmanual.io/usage/models/>`_.
But the difference is that Patito extends pydantic's validation of `singular object instances` to `collections` of the same objects represented as `dataframes`.

Let's take the data presented in `Table 1 <product_table>`_ and represent it as a polars dataframe.

.. code-block:: python

    >>> import polars as pl

    >>> product_df = pl.DataFrame(
    ...     {
    ...         "product_id": [1, 2, 3],
    ...         "name": ["Apple", "Milk", "Ice cubes"],
    ...         "temperature_zone": ["dry", "cold", "frozen"],
    ...     }
    ... )

We can now use :ref:`Product.validate() <Model.validate>` in order to validate the content of our dataframe.

.. code-block:: python

    >>> from project.models import Product
    >>> Product.validate(product_df)
    None

Well, that wasn't really interesting...
The validate method simply returns ``None`` if no errors are found.
It is intended as a guard statement to be put before any logic that requires the data to be valid.
That way you can rely on the data being compatible with the given model schema, otherwise the ``.validate()`` method would have raised an exception.
Let's try this with invalid data, setting the temperature zone of one of the products to ``"oven"``.


.. code-block:: python

    >>> invalid_product_df = pl.DataFrame(
    ...     {
    ...         "product_id": [64, 64],
    ...         "name": ["Pizza", "Cereal"],
    ...         "temperature_zone": ["oven", "dry"],
    ...     }
    ... )
    >>> Product.validate(invalid_product_df)
    ValidationError: 1 validation error for Product
    temperature_zone
      Rows with invalid values: {'oven'}. (type=value_error.rowvalue)

Now we're talking!
Patito allows you to define a single class which validates both singular object instances `and` dataframe collections without code duplication!

.. mermaid::
   :align: center

    %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#FFF5E6', 'secondaryColor': '#FFF5E6' }}}%%
    graph LR;
        pydantic[<code class='literal'>pydantic.BaseModel</code><br /><br />Singular Instance Validation]
        patito[<code class='literal'>patito.Model</code><br /><br />Singular Instance Validation<br />+<br />DataFrame Validation]
        pydantic-->|Same class<br />definition|patito

Patito tries to rely as much as possible on pydantic's existing modelling concepts, naturally extending them to the dataframe domain where suitable.
Model fields annotated with ``str`` will map to dataframe columns stored as ``pl.Utf8``, ``int`` as ``pl.Int8``/``pl.Int16``/.../``pl.Int64``, and so on.
Field types wrapped in ``Optional`` allow null values, while bare types do not.

But certain modelling concepts are not applicable in the context of singular object instances, and are therefore necessarily not part of pydantic's API.
Take ``product_id`` as an example, you would expect this column to be unique across all products and duplicates should therefore be considered invalid.
In pydantic you have no way to express this, but Patito expands upon pydantic in various ways in order to represent dataframe-related constraints.
One of these extensions is the ``unique`` parameter accepted by ``patito.Field``, which allows you to specify that all the values of a given column should be unique.

.. code-block:: python
   :caption: project/models.py::Product

    class Product(pt.Model):
        product_id: int = pt.Field(unique=True)
        name: str
        temperature_zone: Literal["dry", "cold", "frozen"]


The ``patito.Field`` class accepts `the same parameters <https://pydantic-docs.helpmanual.io/usage/schema/#field-customization>`_ as ``pydantic.Field``, but adds additional dataframe-specific constraints documented :ref:`here <Field>`.
If we now use this improved class to validate ``invalid_product_df``, we should receive a new error.

.. code-block:: python

    >>> Product.validate(invalid_product_df)
    ValidationError: 2 validation errors for Product
    product_id
      2 rows with duplicated values. (type=value_error.rowvalue)
    temperature_zone
      Rows with invalid values: {'oven'}. (type=value_error.rowvalue)

Patito has now detected that the given column contains duplicates!
Several more properties and methods are available on ``patito.Model`` as outlined :ref:`here <Model>`; you can for instance generate valid mock dataframes for testing purposes with :ref:`Model.examples() <Model.examples>`.
You can also dynamically construct models with methods such as :ref:`Model.select() <Model.select>`, :ref:`Model.prefix() <Model.prefix>`, and :ref:`Model.join() <Model.join>`.
