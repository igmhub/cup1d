# NumPy-style Docstring Template for cup1d

This document provides a template for documenting cup1d Python code following the NumPy style guide.

## Function Template

```python
def function_name(param1, param2, param3=None):
    """Short description of what the function does.

    Longer description of the function's purpose and behavior.
    Can span multiple lines if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    param3 : type, optional
        Description of param3. Default is None.

    Returns
    -------
    type
        Description of the return value.

    Raises
    ------
    ValueError
        Description of when this error is raised.
    TypeError
        Description of when this error is raised.

    Examples
    --------
    >>> function_name(1, 2)
    3
    >>> function_name(1, 2, 3)
    6

    Notes
    -----
    Any additional implementation details or references.

    References
    ----------
    .. [1] Author name, "Paper title", Journal, Year
    .. [2] Author name, "Book title", Publisher, Year
    """
```

## Class Template

```python
class ClassName:
    """Short description of the class.

    Longer description of what the class does and its purpose.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Attributes
    ----------
    attribute1 : type
        Description of attribute1.
    attribute2 : type
        Description of attribute2.

    Examples
    --------
    >>> obj = ClassName(1, 2)
    >>> obj.method()
    """

    def __init__(self, param1, param2=None):
        """Initialize the class."""
        pass

    def method(self):
        """Short description of the method.

        Parameters
        ----------
        arg : type
            Description of arg.

        Returns
        -------
        type
            Description of return value.
        """
        pass
```

## Type Hints Quick Reference

```python
# Basic types
x: int = 5
y: float = 3.14
z: str = "hello"
flag: bool = True

# Optional types
x: Optional[int] = None
x: int = None  # with default None

# Collections
lst: List[int] = [1, 2, 3]
arr: npt.NDArray[np.float64] = np.array([1.0, 2.0])
dct: Dict[str, float] = {"a": 1.0}

# Union types
x: Union[int, float] = 5
x: Union[int, None] = None

# Tuples
pair: Tuple[int, float] = (1, 2.0)
coords: Tuple[float, float, float] = (1.0, 2.0, 3.0)

# Type aliases (recommended)
Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]
```

## Import Requirements

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Any, Tuple, Union
```

## Common References for cup1d

Add these to your References section:

```python
References
----------
.. [1] Chabanier et al. (2019) - Lyman-alpha forest P1D constraints
.. [2] DESI Collaboration (2024) - DESI Y1 results
.. [3] McDonald et al. (2006) - SDSS Lyman-alpha forest
.. [4] Rogers et al. (2018) - HCD modeling
.. [5] Hui & Gnedin (1997) - IGM thermal history
.. [6] Planck Collaboration (2020) - Planck 2018 results
.. [7] Becker et al. (2013) - IGM thermal constraints
.. [8] Faucher-Giguère et al. (2008) - IGM mean flux
```

## Checklist

- [ ] Module-level docstring with References
- [ ] All public functions have docstrings
- [ ] All classes have docstrings with Parameters
- [ ] Type hints on function signatures
- [ ] Return type annotations
- [ ] Examples in docstrings (optional but recommended)
- [ ] Scientific citations where applicable