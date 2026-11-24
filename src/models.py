from typing import List, Literal
from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    need_retrieval: bool = Field(
        ...,
        description="Whether the question needs retrieval or not",
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="True ONLY if the document contains info that can directly answer the question.",
    )


class BatchRelevanceDecision(BaseModel):
    relevant_indices: List[int] = Field(default_factory=list,
        description="Zero-based indices of documents that directly help answer the question. Exclude unrelated documents.")


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    is_useful: bool = Field(
        ...,
        description="True if the answer is useful and directly addresses the question.",
    )


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against NADRA internal documents.",
    )
