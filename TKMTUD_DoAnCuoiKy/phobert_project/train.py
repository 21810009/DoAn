import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Tên model PhoBERT
MODEL_NAME = "vinai/phobert-base"

# Lớp (class) để xử lý dữ liệu
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Lấy item đã được encode (tokenized)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Thêm nhãn
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- HÀM CHÍNH ĐỂ HUẤN LUYỆN ---
def train_model():
    print("Bắt đầu quá trình huấn luyện...")

    # 1. Tải dữ liệu
    try:
        df = pd.read_csv("data/train.csv")
        texts = df['text'].tolist()
        labels = df['label'].tolist()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file data/train.csv. Hãy tạo file trước.")
        return

    # 2. Tải Tokenizer và Model
    # Tokenizer: Dùng để biến chữ thành số (token) mà model hiểu được
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Model: Tải PhoBERT với 1 lớp phân loại ở trên (num_labels=2 vì ta có 2 loại: Tiêu cực/Tích cực)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3. Tokenize dữ liệu (chuyển chữ thành số)
    print("Đang tokenize dữ liệu...")
    # Chia dữ liệu (ta dùng luôn 1 bộ cho đơn giản, thực tế nên chia train/test)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # Tạo đối tượng Dataset
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    # 4. Thiết lập thông số huấn luyện
    training_args = TrainingArguments(
        output_dir='./model_output',          # Thư mục lưu model
        num_train_epochs=10,               # Số lần lặp lại (epoch)
        per_device_train_batch_size=2,    # Số mẫu mỗi lần train (batch size)
        per_device_eval_batch_size=2,     # Tương tự cho đánh giá
        warmup_steps=100,                 # Số bước "khởi động"
        weight_decay=0.01,                # Kỹ thuật regularization
        logging_dir='./logs',             # Thư mục lưu log
        logging_steps=1,
        eval_strategy="epoch",
       # evaluate_during_training=True,      # Đánh giá model sau mỗi epoch
        save_strategy="epoch",            # Lưu model sau mỗi epoch
        load_best_model_at_end=True,      # Tải model tốt nhất khi kết thúc
    )

    # 5. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 6. Bắt đầu huấn luyện
    print("Bắt đầu huấn luyện model...")
    trainer.train()

    # 7. Lưu model và tokenizer
    print("Huấn luyện hoàn tất. Đang lưu model...")
    trainer.save_model("./model_output")
    tokenizer.save_pretrained("./model_output")
    print(f"Đã lưu model vào thư mục: ./model_output")

if __name__ == "__main__":
    train_model()