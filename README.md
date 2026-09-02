# JetRacer model-only lane steering

Pipeline hiện tại chỉ dùng camera và neural network để dự đoán góc lái. Bird-eye,
lane geometry và pure pursuit đã được loại khỏi đường chạy thực tế.

## Model đang triển khai

- Kiến trúc: PilotNet compact, 252.219 tham số, khởi tạo ngẫu nhiên.
- Input: crop bỏ 32% phần trên ảnh 640×360, sau đó resize RGB thành 200×66.
- Output: một góc lái chuẩn hóa trong `[-1, 1]`.
- ONNX: IR 8, opset 13, khoảng 1 MB; phù hợp ONNX Runtime cũ trên Jetson Nano.
- Controller: lọc thời gian, tăng đáp ứng ở cua, giảm ga theo độ gắt/dao động model.

File triển khai chính:

- `checkpoints/lane_tracker_ir8_opset13.onnx`
- `live_inference.ipynb`
- `src/inference_jetson.py`
- `src/preprocessing_config.py`
- `src/adaptive_controller.py`

## Dataset mới

Dataset nằm tại:

```text
dataset/track_lane_dataset/dataset_steering/session_*/
```

Audit hiện tại có 7.497 frame đang di chuyển từ bốn session, tất cả ảnh 640×360,
không có ảnh hỏng hoặc file trùng. Báo cáo ở `dataset/dataset_report.json`.

Nhãn được làm sạch trong từng session bằng median/mean filter tâm để bỏ các lần
tay cầm nhảy về 0 trong một frame mà không dịch thời điểm cua. Dataset train chỉ
được record ở MANUAL; AUTO chỉ dùng để ghi video và telemetry validation.

## Train lại

Train và giữ một session độc lập để chọn hyperparameter:

```powershell
python -m src.audit_dataset
python -m src.train --device cpu
python -m src.evaluate_onnx
```

Cấu hình được chọn dùng Huber loss, dropout 0,15, horizontal flip và không dùng
recovery-shift giả. Sau khi chọn cấu hình, train final trên toàn bộ bốn session:

```powershell
python -m src.train_final
```

Không lệnh nào ở trên nạp checkpoint cũ hoặc pretrained weights.

## Cập nhật lên Jetson Nano

Từ PowerShell trên PC, thay `<JETSON_IP>` bằng IP của xe:

```powershell
scp live_inference.ipynb jetson@<JETSON_IP>:/home/jetson/Chonf/
scp checkpoints/lane_tracker_ir8_opset13.onnx jetson@<JETSON_IP>:/home/jetson/Chonf/checkpoints/
scp src/inference_jetson.py src/preprocessing_config.py src/adaptive_controller.py src/model.py src/jetracer_fallback.py jetson@<JETSON_IP>:/home/jetson/Chonf/src/
```

Sau khi copy, trong Jupyter chọn **Kernel → Restart Kernel**, rồi chạy các cell
từ trên xuống. AUTO luôn bắt đầu với `Auto max throttle = 0`.

Quy trình chạy thử an toàn:

1. Kê bánh xe khỏi mặt đất, bật AUTO và kiểm tra chiều lái trái/phải.
2. Đặt xe xuống đường, đặt `Auto min throttle = 0.22` và tăng `Auto max throttle`
   từ 0 lên 0.24–0.28.
3. Chỉ tăng thêm 0.02 mỗi lượt khi xe đã qua được toàn bộ cua.
4. Ghi AUTO video + telemetry; không đưa lệnh AUTO trở lại dataset train.

## Kết quả và giới hạn

Ứng viên được chọn trên session chưa dùng để train có MAE 0,258, RMSE 0,314,
tỷ lệ nhầm hướng 21,5% và MAE cua gắt 0,458. Final model sau đó được train lại
trên cả bốn session để có đủ mọi đoạn đường. Vì vậy kết quả `final_seen_session`
chỉ là sanity check, không phải validation độc lập.

Model vẫn chỉ học từ camera đơn và nhãn tay cầm. Safety hiện có thể dừng khi dự
đoán dao động mạnh hoặc FPS quá thấp, nhưng model-only không thể chứng minh xe
đang nằm trong lane. Luôn chạy lượt đầu ở tốc độ thấp và dùng nút dừng khẩn cấp.
