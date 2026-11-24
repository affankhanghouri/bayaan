"""Local BM25 retrieval over bundled NADRA documents; no paid embedding service."""
import math
import re
from collections import Counter
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def tokenize(text):
    aliases = {'renewal': 'renew', 'renewed': 'renew', 'renewing': 'renew',
               'documents': 'document', 'requirements': 'require', 'required': 'require',
               'fees': 'fee', 'children': 'child', 'applications': 'application'}
    return [aliases.get(token, token) for token in re.findall(r'\w+', text.lower())]


class LocalRetriever:
    def __init__(self, k=5, data_dir=None):
        self.k = k
        splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
        self.docs = []
        directory = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent.parent / 'data'
        for path in sorted(directory.glob('*.txt')):
            self.docs.extend(Document(page_content=chunk, metadata={'source': path.name})
                             for chunk in splitter.split_text(path.read_text(encoding='utf-8')))
        self.terms = [Counter(tokenize(doc.page_content)) for doc in self.docs]
        self.df = Counter(term for terms in self.terms for term in terms)
        self.avg = sum(map(lambda c: sum(c.values()), self.terms)) / max(1, len(self.terms))

    def invoke(self, query):
        tokens = set(tokenize(query)) - {'the', 'a', 'is', 'to', 'for', 'of', 'i', 'my', 'how', 'what', 'do', 'can', 'and', 'in', 'are', 'you', 'it', 'an', 'be'}
        scored = []
        for doc, counts in zip(self.docs, self.terms):
            length = sum(counts.values())
            score = sum(math.log(1 + (len(self.docs) - self.df[t] + .5) / (self.df[t] + .5))
                        * counts[t] * 2.5 / (counts[t] + 1.5 * (.25 + .75 * length / self.avg))
                        for t in tokens if counts[t])
            if score > 0:
                scored.append((score, doc))
        return [doc for _, doc in sorted(scored, key=lambda pair: pair[0], reverse=True)[:self.k]]
