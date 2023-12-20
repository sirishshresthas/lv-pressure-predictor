import gradio as gr

from src.core import LVCPredictor

LVCP = LVCPredictor(model_name="xgboost_classifier_model.json")

# Create sliders for the parameters
slider1 = gr.Slider(minimum=0, maximum=10, label="Parameter 1")
slider2 = gr.Slider(minimum=0, maximum=10, label="Parameter 2")
slider3 = gr.Slider(minimum=0, maximum=10, label="Parameter 3")
slider4 = gr.Slider(minimum=0, maximum=10, label="Parameter 4")
slider5 = gr.Slider(minimum=0, maximum=10, label="Parameter 5")

# Create textboxes for the parameters
textbox1 = gr.Textbox(value=5, label="Parameter 1")
textbox2 = gr.Textbox(value=5, label="Parameter 2")
textbox3 = gr.Textbox(value=5, label="Parameter 3")
textbox4 = gr.Textbox(value=5, label="Parameter 4")
textbox5 = gr.Textbox(value=5, label="Parameter 5")

# Combine sliders and textboxes for each parameter
param1_input = textbox1 # gr.Row([slider1, textbox1])
param2_input = textbox2 # gr.Row([slider2, textbox2])
param3_input = textbox3 # gr.Row([slider3, textbox3])
param4_input = textbox4 # gr.Row([slider4, textbox4])
param5_input = textbox5 # gr.Row([slider5, textbox5])

# Create an interface with an image input, parameter sliders, and buttons
iface = gr.Interface(
    fn=LVCP.predict,
    inputs=[param1_input, param2_input, param3_input, param4_input],
    outputs=["text", "image"],
    allow_flagging='never',
    theme=gr.themes.Default(),
    title='LV Capillary Pressure'
)

# Launch the interface
iface.launch(share=True)
