iface = gr.Interface(
    fn=predict_traffic_sign,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs="text",
    live=True,
)

# Launch the Gradio interface
iface.launch()