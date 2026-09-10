from typing import Any
from concrete_fhe_toolkit.math import greater_equal

def transfer(sender_balance: Any, receiver_balance: Any, amount: Any) -> tuple[Any,Any]:
    """Transfers an amount from one account to another securely under encryption.
    
    If the amount is negative or the sender has insufficient balance, the
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
    valid = greater_equal(amount, 0) * greater_equal(sender_balance, amount)
    transferred = amount * valid
    return sender_balance - transferred, receiver_balance + transferred
