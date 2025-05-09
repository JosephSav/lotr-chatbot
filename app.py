import gradio as gr
from rag_utils import load_index, get_top_k_chunks
from sentence_transformers import SentenceTransformer
import requests
import os

HF_API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
HF_HEADERS = {"Authorization": f"Bearer {os.environ['HF_API_TOKEN']}"}

# Load embeddings
index, embeddings, chunks = load_index()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def query_model(prompt):
    try:
        response = requests.post(
            # HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt}
            HF_API_URL,
            headers=HF_HEADERS,
            # json={prompt},
            json=prompt,
        )
        response.raise_for_status()  # Will raise an exception for HTTP error responses
        data = response.json()
        # Try to extract the answer key correctly
        # return response.json()[
        #     "answer"
        # ]  # Adjust if needed based on the actual structure
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def answer_question(user_question, chat_history):
    relevant_chunks = get_top_k_chunks(user_question, embedder, index, chunks, k=3)
    context = "\n".join(relevant_chunks)

    # prompt = f"""You are to answer the question as Gandalf the Grey. You must have some
    # level of character and fantasy fell to your answers, but prioritise clear answers.
    # Use the following context when answering the queries:

    # {context}

    # Question: {user_question}

    # Answer:"""
    # prompt = {
    #     "question": user_question,
    #     "context": f"""
    #                 You are to answer the question as Gandalf the Grey. You must have some
    #                 level of character and fantasy fell to your answers, but prioritise clear answers.
    #                 Use the following context when answering the queries:
    #                 {context}
    #                 """,
    # }
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": f"""
                        You are to answer the question as Gandalf the Grey.
                        Your priorities:
                        20%: Character and fantasy feel
                        80%: Susinct answers to the user questions
                        You may use the following context when answering the question:
                        {context}

                        Question to answer: {user_question}
                        """,
            }
        ],
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
    }

    answer = query_model(prompt)

    # Prepare the response in the required format for Gradio chatbot
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, chat_history


# chatbot = gr.Chatbot(type="messages")
# textbox = gr.Textbox(
#     placeholder="Ask a lore-based question to Gandalf.", label="Your Question"
# )

# chat_interface = gr.Interface(
#     fn=answer_question,
#     inputs=[textbox, chatbot],
#     outputs=[chatbot, chatbot],
#     title="Ask Gandalf: LoTR Lore Chatbot",
#     description="Ask a lore-based question to Gandalf.",
# )
# Set up Gradio interface with unique identifiers for components
with gr.Blocks() as chat_interface:
    chatbot = gr.Chatbot(
        type="messages", label="Gandalf Chat"
    )  # Define the chat component with a label
    textbox = gr.Textbox(
        placeholder="Ask a question...", label="Your Question"
    )  # Define the input text box

    # Connect the function to the inputs and outputs
    textbox.submit(
        answer_question, inputs=[textbox, chatbot], outputs=[chatbot, chatbot]
    )

if __name__ == "__main__":
    chat_interface.launch()
