import gradio as gr

from src.core import LVCPredictor, reset_inputs


class LVCPredictorApp:
    def __init__(self, model_name: str):
        self.LVCP = LVCPredictor(model_name)

    def create_interface(self):

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
                    trpg = gr.Number(
                        value=16, label="TRPG (mmHg)", minimum=0, maximum=150)
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

                # buttons
                with gr.Row():
                    reset = gr.Button('Cancel', variant='secondary')
                    ok = gr.Button("Submit", variant='primary')

            # create a separater line
            gr.HTML("<hr style='margin-top: 50px; margin-bottom: -10px' />")

            with gr.Column():
                gr.Markdown("## Output")

                with gr.Column():
                    pred_box = gr.Markdown(label="Predicted Class (%)", scale=1)
                    plot = gr.Plot(
                        label="Predicted Class Probability", scale=2)

            # reset button
            reset.click(
                reset_inputs,
                inputs=[],
                outputs=[ea,
                         e_prime,
                         e_over_e_prime,
                         trpg,
                         ivc_diam,
                         lv_dim,
                         ef,
                         la_dim,
                         lavi]
            )

            # submit button
            ok.click(
                self.LVCP.predict,
                inputs=[ea,
                        e_prime,
                        e_over_e_prime,
                        trpg,
                        ivc_diam,
                        lv_dim,
                        ef,
                        la_dim,
                        lavi],
                outputs=[pred_box, plot]
            )

        # Launch the app
        demo.launch(share=True)


if __name__ == "__main__":
    app = LVCPredictorApp(model_name="xg_SHAP.json")
    app.create_interface()
