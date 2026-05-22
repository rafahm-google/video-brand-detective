# Gemini Prompting Guide

This document provides a set of best practices and a structured process for creating, reviewing, and enhancing prompts for use with Gemini models. The goal is to produce more accurate, consistent, and relevant results. This guide is based on established prompt engineering techniques.

---

## 1. Core Principles of Effective Prompting

When writing a new prompt, start by incorporating these four key elements. While not all are required for every prompt, they provide a strong foundation.

### a. Persona
Assign a role to the AI. This frames the context and sets the tone for the response. It helps the model adopt a specific viewpoint, style, and level of expertise.

*   **Example:** *"Act as a senior software engineer specializing in Python."*
*   **Example:** *"You are a helpful and witty marketing assistant."*

### b. Task
Clearly and explicitly state what you want the AI to do. Start with a strong action verb. Be specific and avoid ambiguity.

*   **Good:** *"**Summarize** the following text in three bullet points."*
*   **Good:** *"**Generate** a Python script that renames files in a directory."*
*   **Avoid:** *"Can you do something with this text?"*

### c. Context
Provide the necessary background information, data, and examples to guide the model. The more relevant context you provide, the better the output.

*   **Provide Examples (Few-Shot Prompting):** Show the model exactly what you want. This is the most effective way to control the output format and style.
    ```
    **Prompt:**
    Translate the following English phrases to French:
    "Hello" -> "Bonjour"
    "Goodbye" -> "Au revoir"
    "How are you?" -> ?
    ```
*   **Provide Data:** Include the text, code, or data the model needs to work with directly in the prompt.
*   **Reference External Documents:** When using integrated tools (like Gemini in Google Workspace), reference documents directly using the `@` notation (e.g., `Summarize the key findings from @[Project Report Q1]`).

### d. Format
Specify the desired output structure. If you don't, the model will choose for you. Being explicit reduces the need for re-formatting later.

*   **Example:** *"Return the output as a JSON object with the keys 'name', 'email', and 'status'."*
*   **Example:** *"Format the response as a Markdown table."*
*   **Example:** *"Write the answer in a conversational and friendly tone."*

---

## 2. Step-by-Step Prompt Review and Enhancement Process

Use this iterative process to refine existing prompts or improve underperforming ones.

### Step 1: Establish a Baseline (Zero-Shot)
*   **Action:** Run the prompt as-is. If creating a new one, write a simple version using the core principles above without providing explicit examples.
*   **Goal:** Understand the model's default behavior and identify initial gaps in its understanding.

### Step 2: Analyze the Output Critically
*   **Action:** Review the generated response. Ask yourself:
    *   **Accuracy:** Is the information correct?
    *   **Relevance:** Does it fully answer the prompt? Does it include irrelevant information?
    *   **Format:** Is the structure correct (JSON, list, table, etc.)?
    *   **Tone:** Is the style appropriate for the use case?
*   **Goal:** Pinpoint the specific areas where the prompt is failing.

### Step 3: Introduce Examples (Move to Few-Shot)
*   **Action:** If the format, style, or logic is incorrect, edit the prompt to include 1-3 high-quality examples (a "few shots"). The examples should demonstrate exactly what you expect.
*   **Goal:** Steer the model towards the desired output structure and reasoning pattern. This is often the single most effective enhancement.

### Step 4: Refine Specificity and Constraints
*   **Action:** Add more specific instructions or constraints to the prompt.
    *   **Negative Constraints (Use Sparingly):** Tell the model what *not* to do. Example: *"Do not use technical jargon."*
    *   **Positive Instructions (Preferred):** Be more explicit about what you *do* want. Example: *"Explain this concept in simple terms that a 10-year-old could understand."*
    *   **Output Length:** Control the length. Example: *"Summarize in 50 words or less."* or set the `max_tokens` parameter.
*   **Goal:** Reduce ambiguity and narrow the scope of possible outputs.

### Step 5: Experiment with Advanced Techniques
*   **Action:** If the task requires complex reasoning, incorporate an advanced prompting strategy.
    *   **Chain of Thought (CoT):** Add the phrase *"Let's think step by step."* to the end of your prompt. This encourages the model to break down the problem and show its work, which dramatically improves reasoning for math, logic, and multi-step problems.
    *   **Self-Consistency:** For critical reasoning tasks, run the same CoT prompt multiple times with a higher `temperature` (e.g., 0.7) to generate diverse reasoning paths, and then choose the most common answer.
*   **Goal:** Improve the robustness and accuracy of complex reasoning tasks.

### Step 6: Document Your Iterations
*   **Action:** Keep a simple log of your prompt versions and their results. This creates a valuable reference for future work and debugging.
*   **Example Log:**
| Version | Prompt Text | Parameters | Result | Notes |
|---|---|---|---|---|
| 1.0 | `Summarize the report.` | `temp=0.2` | Too long, missed key points. | Initial zero-shot attempt. |
| 1.1 | `Summarize the report in 3 bullet points.` | `temp=0.2` | Better, but still verbose. | Added format constraint. |
| 1.2 | `Act as a busy executive... Summarize...` | `temp=0.2` | Perfect. Concise and to the point. | Added a persona. |
*   **Goal:** Prevent repeating mistakes and build a library of effective prompts.

---

## 3. Key Takeaways and Best Practices Summary

*   **Be Specific and Clear:** The most common cause of bad output is a vague prompt.
*   **Examples are King:** Few-shot prompting is the most reliable way to control output.
*   **Instruct, Don't Just Constrain:** Tell the model what to do, not just what to avoid.
*   **Use "Chain of Thought" for Reasoning:** For any task that isn't simple retrieval or summarization, ask the model to "think step by step."
*   **Iterate:** Prompt engineering is a process of refinement. Don't expect perfection on the first try.
