from enum import Enum

class LoanDecision(str, Enum):
    APPROVE          = "approve"
    REJECT           = "reject"
    REQUEST_DOCS     = "request_docs"
    COUNTER_OFFER    = "counter_offer"