import gradio as gr
# استدعاء ملف مشروعك أو مكتبة التعرف (مثلاً OpenCV أو TensorFlow أو MediaPipe)
# import main 

def predict(image):
    # 1. هنا نمرر الصورة (image) إلى نموذجك
    # result = main.detect_objects(image)
    
    # 2. نرجع النتيجة الفعلية (اسم الشيء أو النسبة)
    return "هنا تظهر نتائج فحص الصورة الفعلية من نموذجك"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="مشروع التعرف على الأشياء"
)

if __name__ == "__main__":
    demo.launch()