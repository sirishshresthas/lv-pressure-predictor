import gradio as gr

from src.core import LVCPredictor, reset_inputs

custom_css = """
.prose > span > p {
    font-size: 1.2rem;
    font-weight: bold;
}
"""


class LVCPredictorApp:
    def __init__(self, model_name: str):
        self.LVCP = LVCPredictor(model_name)

    def create_interface(self):

        with gr.Blocks(css=custom_css) as demo:
            gr.Markdown('## LV Prediction')
            with gr.Column():
                with gr.Row():
                    TTE_EbyA = gr.Number(
                        value="", label="E/A ratio", minimum=0.01, maximum=9.9)
                    TTE_Epr_sep = gr.Number(
                        value="", label="Septal e' (cm/s)", minimum=0.01, maximum=30.0)
                    TTE_EbyEpr_sep = gr.Number(
                        value="", label="Septal E/e' ratio", minimum=0.1, maximum=99.9)

                with gr.Row():
                    TTE_TRPG = gr.Number(
                        value="", label="TRPG (mmHg)", minimum=0, maximum=150)
                    TTE_LAVI = gr.Number(
                        value="", label="LA volume index (ml/m2)", minimum=10, maximum=500)
                    TTE_IVCmax = gr.Number(
                        value="", label="Max IVC diameter (mm)", minimum=1, maximum=50)

                with gr.Row():
                    TTE_Dd = gr.Number(
                        value="", label="LV end-diastolic dimension (mm)", minimum=20, maximum=150)
                    TTE_LVEF = gr.Number(
                        value="", label="LV ejection fraction (%)", minimum=1, maximum=99)
                    TTE_LAd = gr.Number(
                        value="", label="LA dimension (mm)", minimum=10, maximum=150)

                # buttons
                with gr.Row():
                    reset = gr.Button('Cancel', variant='secondary')
                    ok = gr.Button("Submit", variant='primary')

            # create a separater line
            gr.HTML("<hr style='margin-top: 50px; margin-bottom: -10px' />")

            with gr.Column():
                gr.Markdown("## Output")

                with gr.Column():
                    pred_box = gr.Markdown(label="Predicted Class (%)")
                    plot = gr.Plot(
                        label="Explanation")

            # reset button
            reset.click(
                reset_inputs,
                inputs=[],
                outputs=[TTE_EbyA,
                         TTE_Epr_sep,
                         TTE_EbyEpr_sep,
                         TTE_TRPG,
                         TTE_LAVI,
                         TTE_IVCmax,
                         TTE_Dd,
                         TTE_LVEF,
                         TTE_LAd]
            )

            # submit button
            ok.click(
                self.LVCP.predict,
                inputs=[TTE_EbyA,
                        TTE_Epr_sep,
                        TTE_EbyEpr_sep,
                        TTE_TRPG,
                        TTE_LAVI,
                        TTE_IVCmax,
                        TTE_Dd,
                        TTE_LVEF,
                        TTE_LAd],
                outputs=[pred_box, plot]
            )

        # Launch the app
        demo.launch()


if __name__ == "__main__":
    app = LVCPredictorApp(model_name="xg_SHAP.json")
    app.create_interface()
