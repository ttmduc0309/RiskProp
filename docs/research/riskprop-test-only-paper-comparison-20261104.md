# Đối chiếu riêng benchmark test RiskProp với Table 1 paper

Ngày kiểm tra: 04/11/2026. Phần này chỉ dùng metric tính trên **1.344 mẫu test**;
không dùng kết quả validation.

## So sánh kết quả

| Metric Nexar test | RiskProp paper | Benchmark local trên test | Đối chiếu |
|---|---:|---:|---|
| mAUC^0.1 | 0.472 | Chưa được tính trên test | Chưa so sánh được |
| mAUC full | 0.869 | Chưa được tính trên test | Chưa so sánh được |
| mAP | 0.870 | Public 0.875674; Private 0.915007; All 0.895638 | All cao hơn 0.025638 về trị số, nhưng không cùng protocol FAR đã được xác nhận |
| mTTA^0.1 | 0.958 s | 0.868603 s (mọi positive); 0.889789 s (positive có onset quan sát được) | Thấp hơn lần lượt 0.089397 s và 0.068211 s; còn khác cách xử lý censored/miss |

Paper Table 1 báo `0.472 / 0.869 / 0.870 / 0.958 s` cho Nexar. Phần thực nghiệm
nêu metric tuân thủ FAR ≤ 0.1, các horizon 0.5/1.0/1.5 giây và mTTA chỉ tính
cảnh báo sau anomaly onset. Paper nói test Nexar có 1.344 video, nhưng Table 1
không gắn số mAP duy nhất với Public, Private hay một phép gộp cụ thể. [Paper CVPR 2026][paper]

Local benchmark đã chấm đúng 1.344 ID bằng Nexar solution. Public/Private/All
mAP được tính bằng AP trung bình trên ba group của Nexar. Nexar mô tả Public và
Private là hai nửa benchmark/leaderboard và test clips kết thúc tại các mốc
0.5/1.0/1.5 giây trước sự kiện. [Nexar dataset card][nexar]

## Những gì có thể kết luận

- `mAP All = 0.895638` là con số test local gần nhất để đặt cạnh paper `mAP =
  0.870`; chênh số học là **+0.025638**. Tuy nhiên, evaluator Nexar dùng cho
  local tính AP theo group trên toàn đường precision-recall; paper ghi AP dưới
  FAR ≤ 0.1. Vì vậy không kết luận local vượt paper từ chênh lệch này.
- Public `0.875674` và Private `0.915007` là hai điểm split local; paper không
  công bố hai điểm tương ứng nên không có phép ghép trực tiếp.
- Test mTTA local dùng negative-frame threshold grid ở FAR ≤ 0.1, 10 Hz và chỉ
  nhận cảnh báo từ onset trở đi. Có 672 positive, trong đó 16 có onset nằm sau
  đoạn quan sát. Metric gồm tất cả positive gán 16 mẫu censored bằng 0; metric
  evaluable loại 16 mẫu đó, nhưng vẫn tính observable misses là 0. Paper không
  nêu quy tắc riêng cho censored hoặc chi tiết mẫu miss trong mẫu số. Do đó hai
  số local là đối chiếu gần protocol, chưa chứng minh cùng phép tính paper.
- Test benchmark hiện **không có mAUC full hoặc mAUC^0.1**. Các số mAUC được
  ghi ở báo cáo tổng hợp là validation metrics, không được đưa vào so sánh test.

## Kết luận

Benchmark test hiện tại cho ra **mAP Public/Private/All** và **hai biến thể
mTTA**. Trong đó, `mAP All` gần nhất về split tổng thể với paper, còn mTTA là
phép đối chiếu có FAR và timeline. Vì paper/local mAP khác ràng buộc FAR và mTTA
còn khác xử lý một số positive, đây chưa phải so sánh paper-exact. Hai mAUC
trong Table 1 chưa có kết quả test local để đối chiếu.

Artifacts: [`test_results.json`](../../work_dirs/riskprop_nexar_l40_full/benchmark_test_rerun/test_results.json),
[`test_benchmark_report.md`](../../work_dirs/riskprop_nexar_l40_full/benchmark_test_rerun/test_benchmark_report.md).

[paper]: https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf
[nexar]: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
