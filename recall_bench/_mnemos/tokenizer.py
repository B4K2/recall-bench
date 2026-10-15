"""Single-nucleotide (character-level) tokenizer, matching HyenaDNA. No k-mers, no BPE."""

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[UNK]"]
NUCLEOTIDES = ["A", "C", "G", "T", "N"]

VOCAB = SPECIAL_TOKENS + NUCLEOTIDES
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID["[PAD]"]
BOS_ID = TOKEN_TO_ID["[BOS]"]
UNK_ID = TOKEN_TO_ID["[UNK]"]

VOCAB_SIZE = len(VOCAB)


def encode(seq: str) -> list[int]:
    """Uppercase nucleotide string -> token ids. Any non-ACGTN char (stray IUPAC
    ambiguity codes) maps to [UNK]."""
    return [TOKEN_TO_ID.get(ch, UNK_ID) for ch in seq]


def decode(ids) -> str:
    return "".join(ID_TO_TOKEN.get(int(i), "?") for i in ids)
