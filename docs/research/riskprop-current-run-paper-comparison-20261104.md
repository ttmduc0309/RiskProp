# Đối chiếu lần RiskProp hiện tại với kết quả paper

Ngày đối chiếu: 04/11/2026. Nguồn kết quả local là
[`training_benchmark_report.md`](../../work_dirs/riskprop_nexar_l40_full/training_benchmark_report.md)
và benchmark artifacts trong `work_dirs/riskprop_nexar_l40_full/benchmark_test_rerun/`.

## Số liệu paper trên Nexar

Table 1 của RiskProp báo cáo trên Nexar: `mAUC^0.1 = 0.472`, `mAUC = 0.869`,
`mAP = 0.870`, `mTTA^0.1 = 0.958 s`. Paper mô tả metric đánh giá với FAR ≤ 0.1,
ba horizon 0.5/1.0/1.5 giây và cảnh báo TTA chỉ được tính sau anomaly onset.
Paper nêu bộ Nexar có 1.500 video train, test 1.344 video; recipe thực nghiệm
50 epoch, batch 64, 8 NVIDIA A800. [Paper CVPR 2026][paper]

## Kết quả lần chạy local

| Kết quả | Local | Paper Nexar | Đối chiếu |
|---|---:|---:|---|
| Validation mAUC full | 0.8751 | 0.869 | +0.0061 về trị số, nhưng local là validation 300 mẫu, paper là benchmark Nexar; không phải so sánh cùng split |
| Validation standardized partial mAUC, FPR ≤ 0.1 | 0.7403 | 0.472 (`mAUC^0.1`) | Không cùng phép chuẩn hóa; không thể xem chênh lệch là hiệu năng |
| Validation normalized partial mAUC (raw area / 0.1) | 0.5065 | 0.472 (`mAUC^0.1`) | Cùng mục đích giới hạn FAR hơn, nhưng vẫn khác split và chưa xác minh công thức paper trùng phép chuẩn hóa này |
| Validation mAP | 0.8829 | 0.870 | Không cùng protocol: local validation AP không giới hạn FAR; paper gắn FAR ≤ 0.1 vào protocol |
| Test mAP Public | 0.875674 | Paper chỉ công bố một mAP Nexar | Paper không báo tách Public; không đối chiếu trực tiếp |
| Test mAP Private | 0.915007 | Paper chỉ công bố một mAP Nexar | Paper không báo tách Private; không đối chiếu trực tiếp |
| Test mAP All, 1.344 mẫu | 0.895638 | 0.870 | Gần nhất về phạm vi test toàn bộ; local dùng evaluator Nexar AP theo group, paper nêu FAR ≤ 0.1. Chênh trị số +0.025638 không phải delta apples-to-apples |
| Test mTTA, mọi positive | 0.8686 s | 0.958 s | Cùng test Nexar, post-onset và FAR ≤ 0.1; local tính 16 positive bị cắt trước onset là 0 nên chưa hoàn toàn cùng cách xử lý |
| Test mTTA, positive có onset quan sát được | 0.8898 s | 0.958 s | Gần về phạm vi hơn khi bỏ 16 mẫu censored; còn khác biệt mẫu miss và chi tiết threshold/denominator chưa được paper đặc tả đủ |

Các Public/Private/All mAP trong benchmark local được tính bằng cách lấy AP trong
từng group của Nexar rồi trung bình ba group. Nexar chia test Public/Private để
leaderboard và final ranking; evaluator chính thức tính mAP riêng cho từng split.
Paper Table 1 báo một con số mAP duy nhất cho Nexar và không ghi rõ trong bảng đó
nó dùng Public, Private hay phép gộp nào. [Nexar dataset card][nexar]

## Khác biệt recipe huấn luyện

| Thiết lập | Local | Paper |
|---|---|---|
| Epoch | 50 | 50 |
| Dữ liệu train | 1.200; 300 video của pool 1.500 được giữ làm validation | Mô tả Nexar có 1.500 video train; không nêu rõ validation holdout tương đương |
| Hardware | 2 × L40 | 8 × A800 |
| Batch toàn cục | 4 (batch 2/GPU × 2 GPU) | 64 |
| Chọn checkpoint | mAUC full validation; tốt nhất epoch 12 | Paper không nêu metric chọn checkpoint |

Vì batch và số video train khác, lần này xác nhận repo chạy end-to-end và cho
kết quả Nexar đáng kể, nhưng chưa thể kết luận tái lập các số Table 1. README
repo mô tả key `mAUC@` như partial AUC, trong khi code nhánh local hiện lưu best
theo `mAUC_full@`; paper không nói tiêu chí validation/checkpoint selection.
[Official RiskProp repository][repo]

## Kết luận

- **Gần nhất để đặt cạnh paper:** test mTTA `0.8686/0.8898 s` so với `0.958 s`,
  nhưng cần giữ chú thích về 16 mẫu censored và cách tính miss.
- **Không nên tuyên bố mAP vượt paper:** local `All mAP = 0.895638` và paper
  `mAP = 0.870` không cùng protocol FAR/split đã được xác nhận.
- **Validation không phải test:** `mAUC_full = 0.8751` và `mAP = 0.8829` đến từ
  300 video training holdout, không phải 1.344 test videos.
- Kết quả hiện tại là **baseline chạy được**, chưa phải reproduction paper-exact.

[paper]: https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf
[repo]: https://github.com/xingyueye5/RiskProp
[nexar]: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
