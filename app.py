import gradio as gr

from src.core import LVCPredictor

LVCP = LVCPredictor(model_name="xg_SHAP.json")


# Define Gradio output components
output_components = [
    gr.Text(label="Predicted Class (%)"),
    gr.Image(label="Predicted Class Probability"),
]

def debug_function(ea, e_prime, e_over_e_prime, trpg, ivc_diam, lv_dim, ef, la_dim, lavi):
    print(ea, e_prime, e_over_e_prime, trpg, ivc_diam, lv_dim, ef, la_dim, lavi)
    return "Debug output"

def reset_inputs():
    return [0] * 9


with gr.Blocks() as demo:
    gr.Markdown('## LV Prediction')
    with gr.Column():
        with gr.Row():
            ea = gr.Number(value=0.8, label="E/A ratio",
                           minimum=0.01, maximum=9.9)
            e_prime = gr.Number(
                value=6.5, label="septal e' (cm/s)", minimum=0.01, maximum=30.0)
            e_over_e_prime = gr.Number(
                value=10.8, label="septal E/e' ratio", minimum=0.1, maximum=99.9)
        
        with gr.Row():
            trpg = gr.Number(value=16, label="TRPG (mmHg)",
                             minimum=0, maximum=150)
            ivc_diam = gr.Number(
                value=12, label="max IVC diameter (mm)", minimum=1, maximum=50)
            lv_dim = gr.Number(
                value=46, label="LV end-diastolic dimension (mm)", minimum=20, maximum=150)
        
        with gr.Row():
            ef = gr.Number(
                value=55, label="LV ejection fraction (%)", minimum=1, maximum=99)
            la_dim = gr.Number(
                value=37, label="LA dimension (mm)", minimum=10, maximum=150)
            lavi = gr.Number(
                value=32, label="LA volume index (ml/m2)", minimum=10, maximum=500)
            
        with gr.Row():
            reset = gr.Button('Cancel', variant='secondary')
            ok = gr.Button("Submit", variant='primary')


    gr.HTML("<hr style='margin-top: 50px; margin-bottom: -10px' />")
    
    with gr.Column():
        gr.Markdown("## Output")

        with gr.Row():
            pred_box = gr.Text(label="Predicted Class (%)", scale=1)
            plot = gr.Plot(label="Predicted Class Probability", scale=2)


    reset.click(
        reset_inputs,
        inputs=[],
        outputs=[ea, e_prime, e_over_e_prime, trpg, ivc_diam, lv_dim, ef, la_dim, lavi]
    )

    ok.click(
        LVCP.predict,
        inputs=[ea, e_prime, e_over_e_prime, trpg, ivc_diam, lv_dim, ef, la_dim, lavi],
        outputs=[pred_box, plot]
    )

        
            


# ## gradio input components
# demo = gr.Interface(
#     LVCP.predict,
#     [
#         gr.Number(value=0.8, label="E/A ratio", minimum=0.01, maximum=9.9),
#         gr.Number(value=6.5, label="septal e' (cm/s)", minimum=0.01, maximum=30.0),
#         gr.Number(value=10.8, label="septal E/e' ratio", minimum=0.1, maximum=99.9),
#         gr.Number(value=16, label="TRPG (mmHg)", minimum=0, maximum=150),
#         gr.Number(value=12, label="max IVC diameter (mm)", minimum=1, maximum=50),
#         gr.Number(value=46, label="LV end-diastolic dimension (mm)", minimum=20, maximum=150),
#         gr.Number(value=55, label="LV ejection fraction (%)", minimum=1, maximum=99),
#         gr.Number(value=37, label="LA dimension (mm)", minimum=10, maximum=150),
#         gr.Number(value=32, label="LA volume index (ml/m2)", minimum=10, maximum=500)
#     ],
#     allow_flagging='never',
#     outputs=output_components,
#     title="Cardiac Function Prediction",

# )

# Launch the app
demo.launch()
