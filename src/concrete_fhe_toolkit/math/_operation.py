"""User-friendly operation wrappers for bounded math helpers."""

from __future__ import annotations

from typing import Any, Callable


class BoundedOperation:
    """Callable facade that compiles by default and exposes explicit builders."""

    def __init__(
        self,
        name: str,
        make_function: Callable[..., Any],
        compile_function: Callable[..., Any],
        *,
        description: str,
    ) -> None:
        """Initialize the BoundedOperation.
        
        Args:
            name: The name of the operation.
            make_function: Builder function.
            compile_function: Compiler function.
            description: Description of the operation.
            
        Returns:
            None
            
        Example:
            >>> op = BoundedOperation("test", make_fn, compile_fn, description="test")
        """
        self.make = make_function
        self.compile = compile_function
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = (
            f"{description}\n\n"
            "Calling this object compiles a Concrete circuit. Use "
            f"`{name}.make(...)` to get a composable traceable function, or "
            f"`{name}.compile(...)` to be explicit."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compile the operation using the same arguments as `.compile(...)`.
        
        Args:
            *args: Positional arguments for the compiler.
            **kwargs: Keyword arguments for the compiler.
            
        Returns:
            Any: The compiled operation.
            
        Example:
            >>> op(1, 2)
        """
        return self.compile(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the operation.
        
        Returns:
            str: The string representation.
            
        Example:
            >>> repr(op)
        """
        return f"<BoundedOperation {self.__name__}>"
