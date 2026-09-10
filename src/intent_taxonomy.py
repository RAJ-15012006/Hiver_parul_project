"""
intent_taxonomy.py
------------------
Defines the 8-class intent taxonomy for AmazonHelp and provides
keyword-seed helpers used to build the golden evaluation set
and to seed few-shot examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ── Intent definitions ────────────────────────────────────────────────────────

@dataclass
class Intent:
    label: str
    display: str
    description: str
    keywords: List[str]
    few_shot_examples: List[str] = field(default_factory=list)


INTENTS: List[Intent] = [
    Intent(
        label="ORDER_STATUS",
        display="Order Status / Tracking",
        description="Customer asking where their order is, delivery estimate, or tracking link.",
        keywords=["where is my order", "tracking", "shipped", "delivery date",
                  "still not arrived", "when will", "track my", "order status",
                  "hasn't shipped", "estimated delivery"],
        few_shot_examples=[
            "My order was supposed to arrive yesterday and I haven't received it. Where is it?",
            "Can you give me a tracking number for order #112-4567890-1234567?",
            "When will my package arrive? It says 'in transit' for 5 days.",
        ],
    ),
    Intent(
        label="REFUND_RETURN",
        display="Refund / Return",
        description="Customer wants to return an item or get their money back.",
        keywords=["refund", "return", "money back", "send it back", "return policy",
                  "exchange", "reimburse", "credit", "dispute", "charged twice"],
        few_shot_examples=[
            "I want to return this item. It doesn't fit.",
            "I was refunded the wrong amount. Please fix this.",
            "How do I initiate a return on a third-party seller item?",
        ],
    ),
    Intent(
        label="PRODUCT_ISSUE",
        display="Product / Item Issue",
        description="Defective, wrong, or damaged product received.",
        keywords=["broken", "defective", "wrong item", "damaged", "doesn't work",
                  "not working", "faulty", "missing parts", "received wrong",
                  "stopped working", "dead on arrival"],
        few_shot_examples=[
            "The Kindle I received is completely dead. Won't turn on.",
            "I ordered a blue shirt but got a red one.",
            "The packaging was destroyed and the item inside is damaged.",
        ],
    ),
    Intent(
        label="ACCOUNT_ACCESS",
        display="Account / Login Issue",
        description="Login failure, password reset, account locked or hacked.",
        keywords=["can't log in", "login", "password", "account locked", "two-factor",
                  "verify", "sign in", "hacked", "unauthorized access", "reset",
                  "account access", "verification code"],
        few_shot_examples=[
            "I cannot log into my Amazon account. I keep getting 'incorrect password'.",
            "Someone hacked my account. Please lock it immediately.",
            "I'm not receiving the verification code to my email.",
        ],
    ),
    Intent(
        label="DELIVERY_PROBLEM",
        display="Delivery Problem",
        description="Package marked delivered but not received, wrong address, neighbor, etc.",
        keywords=["not received", "says delivered", "missing package", "stolen",
                  "wrong address", "left at neighbor", "not at door", "left outside",
                  "package is missing", "marked delivered but"],
        few_shot_examples=[
            "Amazon says my package was delivered but I never got it.",
            "The driver left my package in the rain and it's destroyed.",
            "My parcel was delivered to the wrong address.",
        ],
    ),
    Intent(
        label="BILLING_CHARGE",
        display="Billing / Unauthorized Charge",
        description="Unexpected charge, double charge, subscription billing issue.",
        keywords=["unauthorized charge", "charged", "billing", "charged twice",
                  "unexpected charge", "statement", "invoice", "credit card",
                  "payment", "overcharged", "charge on my account"],
        few_shot_examples=[
            "I see a charge from Amazon on my credit card that I didn't authorize.",
            "Why was I charged twice for the same order?",
            "I cancelled my subscription but was still billed.",
        ],
    ),
    Intent(
        label="PRIME_MEMBERSHIP",
        display="Prime Membership",
        description="Questions about Prime benefits, cancellation, renewal, price change.",
        keywords=["prime", "prime membership", "cancel prime", "prime video",
                  "prime benefits", "free shipping", "prime day", "prime renewal",
                  "prime cancelled", "annual fee"],
        few_shot_examples=[
            "How do I cancel my Amazon Prime membership?",
            "My Prime free shipping isn't working on this order.",
            "I was charged for Prime renewal but I cancelled it last month.",
        ],
    ),
    Intent(
        label="GENERAL_INQUIRY",
        display="General Inquiry / Other",
        description="Anything else — product questions, recommendations, app issues, etc.",
        keywords=["question", "help", "how do i", "can you explain", "what is",
                  "app not working", "website issue", "customer service",
                  "complaint", "feedback", "recommendation"],
        few_shot_examples=[
            "The Amazon app keeps crashing on my iPhone.",
            "Can you recommend a good laptop under $500?",
            "How do I gift-wrap an order?",
        ],
    ),
]

# Build a lookup dict by label
INTENT_BY_LABEL: Dict[str, Intent] = {i.label: i for i in INTENTS}
LABELS: List[str] = [i.label for i in INTENTS]
LABEL_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABELS)}


def label_description_block() -> str:
    """Format all intents as a numbered list for LLM prompt injection."""
    lines = ["Available intent labels and their meanings:"]
    for idx, intent in enumerate(INTENTS, 1):
        lines.append(f"  {idx}. {intent.label} — {intent.description}")
    return "\n".join(lines)


def few_shot_block(n_per_class: int = 1) -> str:
    """Build a few-shot example block for the classification prompt."""
    lines = ["Classification examples:"]
    for intent in INTENTS:
        ex = intent.few_shot_examples[:n_per_class]
        for e in ex:
            lines.append(f'  Customer: "{e}"')
            lines.append(f'  Intent: {intent.label}')
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(label_description_block())
    print()
    print(few_shot_block())
