import gensim
from gensim.models import Word2Vec
import logging
import os

# Thiết lập logging để thấy tiến trình huấn luyện
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Đường dẫn đến tệp văn bản (đảm bảo tệp này có thật)
FILE_PATH = "lucvantien_da_chuanhoa.txt"
MODEL_SAVE_PATH = "lvt.word2vec.model"

def train_lvt_model(file_path):
    """
    Hàm đọc, tiền xử lý và huấn luyện model Word2Vec từ tệp Lục Vân Tiên.
    """
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy tệp {file_path}")
        return

    print("Đang đọc và tiền xử lý dữ liệu (tokenizing)...")
    
    # 1. Đọc tệp và tiền xử lý (Tokenize)
    # Tệp lucvantien_da_chuanhoa.txt đã được chuẩn hóa, mỗi dòng là một câu
    # Ta sẽ dùng gensim.utils.simple_preprocess (giống như trong notebook 1.1)
    # để tách từ, chuyển về chữ thường và loại bỏ dấu câu (nếu có).
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # simple_preprocess trả về một danh sách các từ
            # Ví dụ: "Trai thời trung hiếu làm đầu" -> ['trai', 'thời', 'trung', 'hiếu', 'làm', 'đầu']
            tokens = gensim.utils.simple_preprocess(line)
            if tokens:
                sentences.append(tokens)

    print(f"Đã xử lý được {len(sentences)} câu.")
    print("Bắt đầu huấn luyện mô hình Word2Vec (Skip-gram)...")

    # 2. Huấn luyện mô hình
    # Chúng ta sử dụng các tham số tương tự như trong notebook 1.1 của anh
    model = Word2Vec(
        sentences=sentences, # Dữ liệu đầu vào
        vector_size=100,     # Kích thước vector: 100 chiều
        window=5,            # Cửa sổ ngữ cảnh: 5 từ trước và 5 từ sau
        min_count=2,         # Chỉ học các từ xuất hiện ít nhất 2 lần
        sg=1,                # sg=1: Dùng thuật toán Skip-gram
        epochs=50,           # Số lần lặp lại (epochs) trên toàn bộ dữ liệu.
        workers=4            # Sử dụng 4 luồng CPU để tăng tốc
    )

    print("Huấn luyện hoàn tất.")
    
    # 3. Lưu model
    model.save(MODEL_SAVE_PATH)
    print(f"Đã lưu model vào: {MODEL_SAVE_PATH}")
    
    return model

def explore_model(model):
    """
    Hàm để khám phá các từ trong mô hình đã huấn luyện.
    """
    print("\n--- Khám phá mô hình ---")
    
    # Do tiền xử lý, "Vân Tiên" sẽ bị tách thành "vân" và "tiên".
    # Chúng ta sẽ kiểm tra các từ này.
    
    # 1. Kiểm tra các từ tương đồng nhất (Most Similar)
    # Đây là các từ có trong truyện Lục Vân Tiên
    words_to_check = ["tiên", "nga", "hiếu", "hung", "trực", "minh"]
    
    for word in words_to_check:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"Các từ tương đồng với '{word}': {similar_words}")
        else:
            print(f"Từ '{word}' không có trong từ vựng (có thể do min_count=2)")
            
    # 2. Thử phép toán tương tự (Analogy)
    # Giống như: king - man + woman = queen
    # Ta thử: "tiên" (Vân Tiên) - "trai" + "gái" -> (kỳ vọng: "nga" hoặc "loan"?)
    try:
        print("\nPhép toán: 'tiên' - 'trai' + 'gái' =>")
        result = model.wv.most_similar(positive=['tiên', 'gái'], negative=['trai'], topn=5)
        print(result)
    except KeyError as e:
        print(f"Không thể thực hiện phép toán, thiếu từ: {e}")

if __name__ == "__main__":
    trained_model = train_lvt_model(FILE_PATH)
    
    if trained_model:
        explore_model(trained_model)