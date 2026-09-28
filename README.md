# cái Repo này để bỏ bài tập môn Machine Learning :3

Bài tập chương 3, viết bằng Python. Gồm 2 phần: tối ưu hàm bằng Gradient Descent
và phân loại điểm dữ liệu bằng Perceptron.

## 📁 Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `matrixCal.py` | Hàm tính đạo hàm đa thức (`derivative`, `grad`, `cost`) |
| `gradient_descent.py` | Nhập đa thức, chạy Gradient Descent (bài 3.26) |
| `run_perceptron.py` | Class `Perceptron` có `fit` và `predict` (bài 3.29) |
| `main_all.py` | Menu chạy bài 3.27, 3.28, 3.29 |
* Cái main_all.py tôi sẽ phát triển thêm (nếu có thời gian) Đơ giản là vì
tôi đang coi trọng việc hoàn thành bài tập đã

## ▶️ Cách chạy

    pip install numpy
    python gradient_descent.py
    python perceptron.py

## 📝 Nội dung từng bài

- **3.26:** f(x) = x² − 4x + 5, x0 = 5, lr = 0.2, 4 bước. Kết quả x tiến dần về 2.
- **3.27, 3.28:** tính wᵀx, xác định nhãn dự đoán, kiểm tra phân lớp sai,
  cập nhật w = w + y·x.
- **3.29:** class Perceptron, `fit` học w trên cả bộ dữ liệu, `predict` dự báo nhãn mới.

## 💡 Mình học được gì

  Thực sự là trước giờ tôi không biết cái gì về code luôn. Nhưng mà bây giờ khi vào chuyên ngành, học về toán, tư duy và trừu tượng thì khiến
tôi cảm thấy rất là thích thú. Tôi code thì ... AI cũng chiếm tầm 50% (hoặc hơn), nhưng tôi sử dụng AI là để học. khi code thì tôi
luôn bảo AI chỉ được gợi ý, còn tôi sẽ là người tư duy, tôi thấy mắc ở đâu thì tôi sẽ hỏi nó.
  Cái mà tôi thấy là học được nhiều nhất đó là ... học code =))) nhưng cũng đồng thời học được cả những bài toán chuyên ngành, 1 công đôi
việc. Tôi cố gắng để hiểu thuật toán, từ đó mới suy luận ra cách để nói cho máy tính chạy (bằng ngôn ngữ tự nhiên). sau khi có ngôn ngữ tự nhiên
tôi bỏ nó vào claude AI để nó hỗ trợ tôi viết hàm trong python. tôi khi đó cũng hiểu được từng câu lệnh trong python luôn...cũng vui
