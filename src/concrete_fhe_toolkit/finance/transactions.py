from typing import Any
from concrete_fhe_toolkit.math import greater_equal

def transfer(sender_balance: Any, receiver_balance: Any, amount: Any) -> tuple[Any,Any]:
    """Transfers an amount from one account to another."""
    is_balance_enough = greater_equal(sender_balance,amount)
    amount *= is_balance_enough 
    sender_balance -= amount
    receiver_balance += amount

    return sender_balance,receiver_balance