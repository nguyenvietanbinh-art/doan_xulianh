import gradio as gr
import summarize_module1
import os
import traceback

# --- 1. TÙY CHỈNH THEME ---
try:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )
except:
    theme = None

# --- 2. HÀM XỬ LÝ ---
def summarize_wrapper(video_input):
    if video_input is None:
        return "⚠️ Cảnh báo: Vui lòng tải video lên trước!", None

    try:
        if os.path.exists("summary_output.mp4"):
            os.remove("summary_output.mp4")

        result = summarize_module1.summarize_video(video_input, "summary_output.mp4", gt_intervals=None)
        
        output_path = result["output_path"]
        keyframes_count = len(result['keyframes'])
        comp_ratio = result.get('compression_ratio', 0)
        recall = result.get('recall')

        if not os.path.exists(output_path):
            return "❌ Lỗi: Không tạo được file video.", None

        # --- ĐỊNH DẠNG HIỂN THỊ KẾT QUẢ ---
        
        ratio_str = f"{comp_ratio:.2f} lần" if comp_ratio else "N/A"
        
        recall_str = f"{recall*100:.2f}%" if recall is not None else "N/A (Chưa có dữ liệu mẫu)"

        status_msg = (
            f" <b>Xử lý thành công!</b><br><br>"
            f" <b>Kết quả chi tiết:</b><br>"
            f"• Số đoạn sự kiện (Shots): <b>{keyframes_count}</b><br>"
            f"• Tỷ lệ nén (Compression Ratio): <b>{ratio_str}</b><br>"
            f"• Độ bao phủ (Recall): <b>{recall_str}</b><br><br>"
            f" Đường dẫn file: <code>{output_path}</code><br>"
            f" <i>Gợi ý: Video tóm tắt bao gồm các đoạn clip quan trọng được ghép lại.</i>"
        )
        
        return status_msg, output_path

    except Exception as e:
        traceback.print_exc()
        return f"❌ Lỗi nghiêm trọng: {str(e)}", None

# --- 3. XÂY DỰNG GIAO DIỆN ---
block_kwargs = {"css": "style.css", "title": "Video Summarizer"}
if theme:
    block_kwargs["theme"] = theme

with gr.Blocks(**block_kwargs) as demo:
    
    with gr.Row(elem_classes="header-text"):
        gr.HTML("<h1>Tóm Tắt Video</h1>")
    
    gr.Markdown("---")
    
    with gr.Accordion("Hướng dẫn sử dụng", open=False):
        gr.Markdown("1. Tải video lên.\n2. Bấm nút bắt đầu.\n3. Xem kết quả và các chỉ số đánh giá.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Video Đầu Vào")
            video_in = gr.Video(label="Input", sources=["upload"], height=300)
            btn = gr.Button("Bắt đầu Tóm tắt", variant="primary", size="lg")
            
            gr.Markdown("### 📝 Báo Cáo Kết Quả")
            status_out = gr.HTML(value="<div class='status-box'>Hệ thống sẵn sàng...</div>")

        with gr.Column(scale=1):
            gr.Markdown("### 🎬 Video Kết Quả")
            video_out = gr.Video(label="Output", interactive=False, height=300)

    gr.Markdown("---")
    gr.Markdown("<div style='text-align: center; color: gray;'>Đồ án Xử Lý Ảnh | 2025 | Nguyễn Viết An Bình | Cao Trọng Gia Cường</div>")

    def formatting_wrapper(vid):
        msg, path = summarize_wrapper(vid)
        # Bọc vào div status-box để nhận CSS
        formatted_msg = f"<div class='status-box'>{msg}</div>"
        return formatted_msg, path

    btn.click(
        fn=formatting_wrapper,
        inputs=video_in,
        outputs=[status_out, video_out],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["."])