"""
app.py
Owner: YOU (interface + evaluation).

This is the project's user interface: a web page (built with the Gradio
library) where the user uploads an image, types what to look for, and sees
the result with boxes drawn on top.

No HTML/CSS/JavaScript needed: Gradio generates the web interface on its
own from the Python code below.
"""

import os

# Avoid background analytics requests that fail behind strict proxies.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Keep matplotlib cache inside the project (writable in locked environments).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import gradio as gr
import numpy as np
import detector
import render
import video as video_pipeline


# List of models available in the dropdown menu.
# Start with zero-shot inference (YOLO-World). Keep stub as a fallback
# while wiring/troubleshooting.
MODEL_OPTIONS = [
    "YOLO-World (zero-shot)",
    "Fine-tuned (best.onnx)",
    "stub (fake data, for testing)",
]

# Class names the fine-tuned model was trained on. Loaded once at startup so
# the dropdown always offers words the model actually recognises.
FINE_TUNED_CLASSES = detector.get_fine_tuned_class_names()


def _is_fine_tuned(model_choice):
    return "fine-tuned" in (model_choice or "").lower()


def _toggle_query_inputs(model_choice):
    """Show the free-text box for zero-shot, the class dropdown for fine-tuned."""
    fine_tuned = _is_fine_tuned(model_choice)
    return (
        gr.update(visible=not fine_tuned),  # free-text query box
        gr.update(visible=fine_tuned),      # fine-tuned class dropdown
    )


def run_pipeline(image, query, class_filter, confidence, model_choice):
    """
    This function is called every time the user presses the "Find objects"
    button. It wires all the pieces together:
        1. takes the uploaded image
        2. calls detector.detect() to find the objects
        3. calls render.draw() to draw them on top of the image
        4. returns the final image to display
    """
    if image is None:
        return None, "Upload an image first."

    fine_tuned = _is_fine_tuned(model_choice)

    if not fine_tuned and not query.strip():
        return None, "Type what to look for (e.g. 'person', 'car')."

    # Fine-tuned model filters by the classes picked in the dropdown; the
    # zero-shot / stub models use the free-text query.
    detect_query = class_filter if fine_tuned else query

    # Gradio gives a PIL image when `type="pil"`. `render.draw()` expects an
    # HxWx3 NumPy RGB array.
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    detections = detector.detect(image_rgb, query=detect_query, confidence=confidence, model_choice=model_choice)
    result_image = render.draw(image_np, detections, conf_threshold=confidence)


    if fine_tuned:
        if class_filter:
            status = f"Found {len(detections)} objects matching {', '.join(class_filter)}."
        else:
            status = f"Found {len(detections)} objects across all trained classes."
    else:
        status = f"Found {len(detections)} objects matching '{query}'."
    return result_image, status


def run_video_pipeline(video_file, query, class_filter, confidence, model_choice, sample_rate):
    """Video path: upload -> detect/render -> annotated mp4."""
    fine_tuned = _is_fine_tuned(model_choice)
    if not fine_tuned and not query.strip():
        return None, "Type what to look for (e.g. 'person', 'car')."
    detect_query = class_filter if fine_tuned else query
    return video_pipeline.process_video(
        input_path=video_file,
        query=detect_query,
        confidence=confidence,
        model_choice=model_choice,
        sample_rate=sample_rate,
    )


# --- Building the interface ---
with gr.Blocks(title="DroneFind") as demo:
    gr.Markdown("# DroneFind")
    gr.Markdown(
        "Upload an image, describe what to look for, and get the result "
        "with the matching objects circled."
    )

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Image")
                    query_input = gr.Textbox(
                        label="What to look for",
                        placeholder="e.g. 'person', 'car'",
                        visible=True,
                    )
                    class_input = gr.Dropdown(
                        choices=FINE_TUNED_CLASSES,
                        value=[],
                        multiselect=True,
                        label="Classes to keep (leave empty for all)",
                        visible=False,
                    )
                    confidence_input = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                        label="Confidence threshold",
                    )
                    model_input = gr.Dropdown(
                        choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model"
                    )
                    submit_button = gr.Button("Find objects", variant="primary")

                with gr.Column():
                    image_output = gr.Image(label="Result")
                    status_output = gr.Textbox(label="Status", interactive=False)

            model_input.change(
                fn=_toggle_query_inputs,
                inputs=model_input,
                outputs=[query_input, class_input],
            )

            submit_button.click(
                fn=run_pipeline,
                inputs=[image_input, query_input, class_input, confidence_input, model_input],
                outputs=[image_output, status_output],
            )

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input video")
                    video_query_input = gr.Textbox(
                        label="What to look for",
                        placeholder="e.g. 'person', 'car'",
                        visible=True,
                    )
                    video_class_input = gr.Dropdown(
                        choices=FINE_TUNED_CLASSES,
                        value=[],
                        multiselect=True,
                        label="Classes to keep (leave empty for all)",
                        visible=False,
                    )
                    video_confidence_input = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                        label="Confidence threshold",
                    )
                    video_model_input = gr.Dropdown(
                        choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Model"
                    )
                    sample_rate_input = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=5,
                        step=1,
                        label="Process every Nth frame",
                    )
                    video_submit_button = gr.Button("Process video", variant="primary")

                with gr.Column():
                    video_output = gr.Video(label="Annotated video")
                    video_status_output = gr.Textbox(label="Status", interactive=False)

            video_model_input.change(
                fn=_toggle_query_inputs,
                inputs=video_model_input,
                outputs=[video_query_input, video_class_input],
            )

            video_submit_button.click(
                fn=run_video_pipeline,
                inputs=[
                    video_input,
                    video_query_input,
                    video_class_input,
                    video_confidence_input,
                    video_model_input,
                    sample_rate_input,
                ],
                outputs=[video_output, video_status_output],
            )


if __name__ == "__main__":
    # `show_api=False` avoids a Gradio 4.44 schema bug in some envs.
    # `share=True` is needed when localhost is not directly reachable.
    demo.launch(show_api=False, share=True)
