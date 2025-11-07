
# import gradio as gr
# import requests
# import io
# from PIL import Image

# API_URL = "http://127.0.0.1:8000/query"
# IMG_URL = "http://127.0.0.1:8000/images/"

# def chat(message, history):
#     # Append user message
#     history = history or []
#     history.append([message, None])  # User message

#     try:
#         # Send to backend
#         r = requests.post(API_URL, json={"query": message}, timeout=15)
#         if r.status_code != 200:
#             bot_msg = f"Backend error {r.status_code}: {r.text}"
#             history[-1][1] = bot_msg
#             yield history, None
#             return

#         data = r.json()
#         bot_response = data.get("response", "No response.")
#         products = data.get("products", [])[:2]

#         # Build bot message
#         for p in products:
#             bot_response += f"\n\n**{p['name']}** – Rs{p['cost']:.2f} | {p['rating']}/5\n{p['review']}"

#         # Update last message with bot response
#         history[-1][1] = bot_response

#         # Yield updated history + image (if any)
#         if products and products[0].get("image"):
#             fname = products[0]["image"].split("/")[-1]
#             ir = requests.get(IMG_URL + fname, timeout=10)
#             if ir.status_code == 200:
#                 img = Image.open(io.BytesIO(ir.content))
#                 yield history, gr.Image(value=img, label=products[0]["name"])
#             else:
#                 yield history, None
#         else:
#             yield history, None

#     except Exception as e:
#         history[-1][1] = f"Error: {str(e)}"
#         yield history, None


# with gr.Blocks(title="E-Com Chat") as demo:
#     gr.Markdown("# E-Commerce RAG Chatbot")
#     chatbot = gr.Chatbot(height=200)
#     txtbox = gr.Textbox(placeholder="Ask about a product…", lines=2, label="Query")
#     image_output = gr.Image(label="Product Image", visible=False)

#     # Buttons
#     submit_btn = gr.Button("Submit")
#     clear_btn = gr.Button("Clear")

#     # Submit actions
#     submit_btn.click(chat, [txtbox, chatbot], [chatbot, image_output])
#     txtbox.submit(chat, [txtbox, chatbot], [chatbot, image_output])

#     # Clear
#     clear_btn.click(
#         lambda: ([], None, None),
#         None,
#         [chatbot, image_output, txtbox]
#     )

#     gr.Examples(
#         [["rating of denim jeans?"], ["electronics under 200rs"], ["best sports gear"]],
#         inputs=txtbox
#     )

# if __name__ == "__main__":
#     print("Starting Gradio UI... Go to http://127.0.0.1:7860")
#     demo.launch(server_name="127.0.0.1", server_port=7860, share=False)






import gradio as gr
import requests
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/query"
IMG_URL = "http://127.0.0.1:8000/images/"

def chat(message, history):
    history = history or []
    history.append([message, None])

    try:
        r = requests.post(API_URL, json={"query": message}, timeout=15)
        if r.status_code != 200:
            history[-1][1] = f"Error {r.status_code}"
            yield history, None
            return

        data = r.json()
        answer = data["response"]
        product = data["products"][0]

       
        name = product["name"]
        cost = product["cost"]
        rating = product["rating"]
        review = product["review"]
        image_path = product.get("image")

        bot_msg = f"**{name}** – ₹{cost:.2f} | {rating}/5\n{review}"

        
        final_msg = f"{answer}\n\n{bot_msg}"

        history[-1][1] = final_msg

       
        if image_path:
            fname = image_path.split("/")[-1]
            ir = requests.get(IMG_URL + fname, timeout=10)
            if ir.status_code == 200:
                img = Image.open(io.BytesIO(ir.content))
                yield history, gr.Image(value=img, label=name, height=300)
            else:
                yield history, None
        else:
            yield history, None

    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield history, None


with gr.Blocks(title="E-Com Bot") as demo:
    gr.Markdown("E-Commerce Product Assistant")
    

    chatbot = gr.Chatbot(height=300)
    txtbox = gr.Textbox(placeholder="e.g. rating of denim jeans", lines=2, label="Your Question")
    image_out = gr.Image(label="Product Image", height=300, visible=False)

    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    submit_btn.click(chat, [txtbox, chatbot], [chatbot, image_out])
    txtbox.submit(chat, [txtbox, chatbot], [chatbot, image_out])
    clear_btn.click(lambda: ([], None, None), None, [chatbot, image_out, txtbox])

    gr.Examples([
        ["what is the rating of denim jeans?"],
        ["price of bluetooth speaker"],
        ["review of yoga mat"],
        ["show me the smart watch"]
    ], inputs=txtbox)

if __name__ == "__main__":
    print("Starting UI → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)