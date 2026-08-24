from typing import Any
from concrete_fhe_toolkit.math import greater_equal

def transfer(sender_balance: Any, receiver_balance: Any, amount: Any) -> tuple[Any,Any]:
    """Transfers an amount from one account to another securely under encryption.
    
    If the sender has insufficient balance (sender_balance < amount), the 
    transfer is silently cancelled (amount becomes 0) without leaking 
    information about the failure.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.transactions import transfer
        
        # Inside an FHE circuit
        # enc_new_sender_bal, enc_new_receiver_bal = transfer(
        #     enc_sender_bal, enc_receiver_bal, enc_transfer_amount
        # )
        ```
    """
    is_balance_enough = greater_equal(sender_balance,amount)
    amount *= is_balance_enough 
    sender_balance -= amount
    receiver_balance += amount

    return sender_balance,receiver_balance