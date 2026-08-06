from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# model declaration
model = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are an expert AI assistant.
Be concise and technical.
Give code examples where a coding related questions are asked.
"""

EVAL_SYSTEM = """
You are an expert evaluator of AI responses.
Score the given response on three dimensions.
Return ONLY a JSON object - no markdown, no explanation.
"""

EVAL_HUMAN = """
Question : {question}
Response : {answer}

Score each dimension from 0.0 to 1.0:
  relevance   : Does the response directly answer the question?
  coherence   : Is the response well-structured and logical?
  conciseness : Is the response appropriately brief without missing key points?
  feedback    : One sentence explaining the main strength or weakness.

Return ONLY this JSON:
{{"relevance": 0.0, "coherence": 0.0, "conciseness": 0.0, "feedback": "..."}}
"""

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM),
    ("human",  EVAL_HUMAN),
])

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human",  "{input}"),
])


def get_chain(api_key):
    """Build the chat chain using a user-supplied Groq API key."""
    llm = ChatGroq(model=model, temperature=0.7, max_retries=3, api_key=api_key)
    return prompt | llm | StrOutputParser()


def get_eval_chain(api_key):
    """Build the evaluation chain using a user-supplied Groq API key."""
    eval_llm = ChatGroq(model=model, temperature=0.7, max_retries=3, api_key=api_key)
    return eval_prompt | eval_llm | JsonOutputParser()
