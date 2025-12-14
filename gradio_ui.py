import gradio as gr
import summarize_module
import os

def summarize_wrapper(video_input):
    try:
        result = summarize_module.summarize_video(video_input, "summary_output.mp4")

        
        with open(result["output_path"], "rb") as f:
            video_bytes = f.read()

        status = f"✓ Tóm tắt thành công!\nKeyframes: {result['keyframes']}"

        return status, (video_bytes, "video/mp4")

    except Exception as e:
        return f"❌ Lỗi: {e}", None


with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Tóm tắt video bằng OpenCV")

    video_in = gr.Video(label="Chọn video đầu vào")
    status_out = gr.Textbox(label="Kết quả")
    video_out = gr.Video(label="Video đã tóm tắt")

    btn = gr.Button("Tóm tắt")

    btn.click(
        fn=summarize_wrapper,
        inputs=video_in,
        outputs=[status_out, video_out],
    )

demo.launch()
