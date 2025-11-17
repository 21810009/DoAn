import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Thư mục chứa model đã huấn luyện
MODEL_PATH = "./model_output"
LABELS = ["Tiêu cực", "Tích cực"]  # Nhãn 0 và 1 tương ứng


def predict_sentiment(text, tokenizer, model):
    # 1. Tokenize (biến câu thành số)
    inputs = tokenizer(
        text,
        return_tensors="pt",  # Trả về dạng PyTorch tensor
        truncation=True,
        padding=True,
        max_length=128
    )

    # 2. Chạy dự đoán
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm bộ nhớ
        outputs = model(**inputs)

    # 3. Lấy kết quả
    logits = outputs.logits
    # Tìm nhãn có xác suất cao nhất (argmax)
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # 4. Trả về nhãn dự đoán
    return LABELS[predicted_class_id]


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    print("Đang tải model và tokenizer...")
    # Tải tokenizer và model đã lưu một lần duy nhất
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"Lỗi: Không tìm thấy model tại '{MODEL_PATH}'.")
        print("Bạn cần chạy 'python train.py' trước để huấn luyện và lưu model.")
        exit()  # Thoát nếu không tìm thấy model

    # Đặt model ở chế độ "eval" (không huấn luyện)
    model.eval()
    print("Model đã sẵn sàng để dự đoán.")
    print("Nhập câu để dự đoán cảm xúc (gõ 'thoat' để thoát chương trình).")

    while True:
        user_input = input("Nhập câu của bạn: ")
        if user_input.lower() == 'thoat':
            print("Kết thúc chương trình.")
            break

        if not user_input.strip():  # Kiểm tra nếu người dùng chỉ nhập khoảng trắng
            print("Vui lòng nhập một câu có nghĩa.")
            continue

        result = predict_sentiment(user_input, tokenizer, model)
        print(f"==> Dự đoán: {result}\n")