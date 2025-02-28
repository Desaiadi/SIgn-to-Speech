# Evaluation Metrics Documentation

## Overview
The `Evaluation_Metrics` class provides an automated framework for evaluating the quality of responses generated from retrieved contexts. It implements three key evaluation metrics:

1. **Context Precision Without Reference** – Assesses the relevance of retrieved contexts in relation to the user query.
2. **Response Relevancy** – Measures how well the generated response aligns with the user’s intent.
3. **Faithfulness** – Evaluates whether the response is factually accurate based on the retrieved contexts.

These metrics utilize large language models (LLMs) and vector embeddings to ensure that responses are both contextually appropriate and factually reliable.

---

## 1. Context Precision Without Reference

### Definition
Context Precision Without Reference measures the proportion of retrieved contexts that are relevant to the given query. This metric evaluates whether the retrieved contexts contain useful information for generating an appropriate response. It does not assess the factual accuracy of the response, only the relevance of the retrieved information.

### Calculation
Context Precision at a given rank (K) is calculated as follows:

\[\text{Context Precision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times v_k)}{\text{Total number of relevant items in the top } K}\]

Where:
- **K** represents the number of retrieved context chunks.
- **v_k** is an indicator of whether the retrieved chunk at position k is relevant, where 1 indicates relevance and 0 indicates irrelevance.
- **Precision@k** is defined as:

\[\text{Precision@k} = \frac{\text{True Positives at rank } k}{\text{True Positives at rank } k + \text{False Positives at rank } k}\]

A higher Context Precision score indicates that a greater proportion of retrieved contexts were relevant to answering the query.

### Implementation
The function to compute Context Precision is as follows:

```python
async def calculate_context_precision(self):
    """Evaluates Context Precision Without Reference."""
    sample = SingleTurnSample(
        user_input=self.user_input,
        response=self.response,
        retrieved_contexts=self.retrieved_context,
    )
    score = await self.context_precision.single_turn_ascore(sample)
    return {"Context Precision": score}
```

### Example
#### User Question:  
*"What is the capital of France?"*

#### Retrieved Contexts:
1. "Paris is the capital of France." (Relevant)
2. "France is in Western Europe." (Not directly relevant)
3. "Germany's capital is Berlin." (Irrelevant)

If only one out of three retrieved contexts is relevant, the Context Precision score will be low.

---

## 2. Response Relevancy

### Definition
Response Relevancy measures how well the generated response aligns with the user input. A higher score indicates that the response appropriately addresses the user’s question, while a lower score suggests that the response is incomplete or contains unrelated details.

### Calculation
Response Relevancy is calculated using cosine similarity between the embedding of the user input and the embeddings of artificial questions generated based on the response. The final score is computed as:

\[\text{Response Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(\text{similarity}(E_{g_i}, E_{o}))\]

Where:
- **E_{g_i}** represents the embedding of the i-th generated question.
- **E_{o}** represents the embedding of the user input.
- **N** is the number of generated questions, with a default value of three.

A higher Response Relevancy score indicates that the response effectively aligns with the intent of the original question.

### Implementation
The function to compute Response Relevancy is as follows:

```python
async def calculate_response_relevancy(self):
    """Evaluates Response Relevancy using OpenAI Embeddings."""
    sample = SingleTurnSample(
        user_input=self.user_input,
        response=self.response,
        retrieved_contexts=self.retrieved_context,
    )
    score = await self.response_relevancy.single_turn_ascore(sample)
    return {"Response Relevancy": score}
```

### Example
#### User Question:  
*"Where is France and what is its capital?"*

#### Low Relevance Answer:
*"France is in Western Europe."*
- The response does not provide complete information.
- The generated artificial questions might include:
  - "Where is France located?"
  - "What region is France in?"
  - "In which part of Europe is France?"

Since none of these questions focus on the capital, the Response Relevancy score is low.

#### High Relevance Answer:
*"France is in Western Europe, and Paris is its capital."*
- The generated questions might include:
  - "Where is France located?"
  - "What is the capital of France?"
  - "Which country has Paris as its capital?"

Since these questions fully cover the query’s intent, the Response Relevancy score is high.

---

## 3. Faithfulness

### Definition
Faithfulness evaluates whether the generated response is factually accurate according to the retrieved context. A high Faithfulness score indicates that all claims in the response are directly supported by the retrieved context, whereas a low score suggests the presence of hallucinated or unsupported claims.

### Calculation

\[\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}\]

Where:
- A **claim** is a factual statement within the response.
- A **claim is supported** if it can be directly inferred from the retrieved context.

### Implementation
The function to compute Faithfulness is as follows:

```python
async def calculate_faithfulness(self):
    """Evaluates Faithfulness."""
    sample = SingleTurnSample(
        user_input=self.user_input,
        response=self.response,
        retrieved_contexts=self.retrieved_context,
    )
    score = await self.faithfulness.single_turn_ascore(sample)
    return {"Faithfulness": score}
```

### Example
#### User Question:  
*"Where and when was Einstein born?"*

#### High Faithfulness Answer:
*"Einstein was born in Germany on March 14, 1879."*
- Both claims are supported by the retrieved context.
- Faithfulness Score = 1.0.

#### Low Faithfulness Answer:
*"Einstein was born in Germany on March 20, 1879."*
- The claim regarding Germany is correct.
- The claim about the birthdate is incorrect.
- Faithfulness Score = 0.5.

---

## Comparison of the Three Metrics

| **Metric** | **What It Evaluates** | **Key Focus** | **A Low Score Means...** |
|------------|----------------------|---------------|--------------------------|
| **Context Precision** | Relevance of retrieved contexts | Ensures retrieved data is useful for generating the response | Retrieved contexts are not relevant to the query |
| **Response Relevancy** | How well the response aligns with the user’s intent | Checks if the response directly answers the question | The response is incomplete or off-topic |
| **Faithfulness** | Whether the response is factually correct | Ensures the response is free from hallucinations | The response contains unsupported or false claims |

This documentation provides a comprehensive guide to understanding and implementing these evaluation metrics.

