from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_exa import ExaSearchRetriever
from langchain_openai import ChatOpenAI

retriever = ExaSearchRetriever(k=3, highlights=True)

document_prompt = PromptTemplate.from_template(
    """
<source>
    <url>{url}</url>
    <highlights>{highlights}</highlights>
</source>
"""
)

document_chain = (
    RunnableLambda(
        lambda document: {
            "highlights": document.metadata["highlights"],
            "url": document.metadata["url"],
        }
    )
    | document_prompt
)

retrieval_chain = (
    retriever | document_chain.map() | (lambda docs: "\n".join([i.text for i in docs]))
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research assistant. You use xml-formatted context to research people's questions.",
        ),
        (
            "human",
            """
Please answer the following query based on the provided context. Please cite your sources at the end of your response.:
     
Query: {query}
---
<context>
{context}
</context>
""",
        ),
    ]
)

llm = ChatOpenAI()

chain = (
    RunnableParallel(
        {
            "query": RunnablePassthrough(),
            "context": retrieval_chain,
        }
    )
    | generation_prompt
    | llm
).with_types(input_type=str)
