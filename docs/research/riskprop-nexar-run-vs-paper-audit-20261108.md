# Vì sao lượt RiskProp Nexar hiện tại khác kết quả paper?

Ngày nghiên cứu: 08/11/2026  
Phạm vi: run full-data RiskProp trên Nexar trong `work_dirs/riskprop_nexar_l40_full`, đối chiếu paper CVPR 2026 và repository RiskProp chính thức. Không chạy lại training/benchmark và không sửa logic model.

## Kết luận ngắn

**Có khác biệt về training config đủ lớn để làm kết quả thay đổi, nhưng chưa thể quy toàn bộ chênh lệch cho config.** Khác biệt rõ nhất là global batch `4` thay vì `64`, và local run dùng 1.200 video để train sau khi tách 300 video làm validation. Lựa chọn checkpoint cũng khác/không được paper công bố. Các yếu tố này có thể làm thay đổi tối ưu hóa và mô hình cuối.

Đồng thời, một số metric local **không cùng protocol** với số paper: mAP local là Nexar evaluator theo group và không giới hạn FAR như protocol paper mô tả; mAUC local có cả full ROC và hai biến thể partial; mTTA local là một phép tính full-timeline có xử lý video bị censor. Vì vậy, một phần “khác số” đến từ phép đo, không nhất thiết là model.

Ngoài ra, đối chiếu mô tả toán học paper với `taa/models.py` cho thấy một số lệch về cách áp dụng auxiliary loss trong code upstream: FFR vẫn tính trên negative video; trung bình risk dùng trong AMC tính trên cả batch thay vì chỉ positive video; paper nói collision anchor được gán trọng số cao hơn nhưng config đang dùng `pos_weight=1` và không có trọng số riêng cho collision anchor. Các điểm này nằm trong code upstream mà checkout dựa trên, không chỉ do cấu hình phần cứng của run.

## Kết quả đang so sánh

| Metric | Paper Nexar | Repo `All` | Lưu ý về khả năng so sánh |
|---|---:|---:|---|
| mAP | 0,870 | 0,895638 | Local là Nexar official grouped AP; paper nói AP dưới FAR ≤ 0,1. Không đồng nhất đã được xác nhận. |
| Full mAUC | 0,869 | 0,905417 | Local là trung bình AUC đầy đủ theo 3 group; paper mô tả đánh giá theo các horizon và FAR constraint. Cần evaluator paper-exact trước khi kết luận delta hiệu năng. |
| mAUC@FAR 0,1 | 0,472 | 0,497767 normalized raw pAUC; 0,735667 sklearn standardized pAUC | Paper không mô tả normalization đủ rõ để khẳng định biến thể nào bằng công thức đã dùng cho 0,472. Không so sánh standardized `0,735667` trực tiếp với `0,472`. |
| mTTA@FAR 0,1 | 0,958 s | 0,868603 s all positives; 0,889789 s evaluable positives | Local tính từ chuỗi điểm full-timeline, 16 positive bị censor; paper chỉ tính alarm sau anomaly onset và mô tả mean TTA của true-positive alarms. Mẫu số/threshold protocol chưa chắc trùng. |

Số paper đến từ Bảng 1 của paper CVPR chính thức. Local `All` và ghi chú về evaluator lấy từ [test report](../../work_dirs/riskprop_nexar_l40_full/benchmark_test_full_mauc_20261104/test_benchmark_report.md) và [training/benchmark summary](../../work_dirs/riskprop_nexar_l40_full/training_benchmark_report.md).

## Đối chiếu cấu hình huấn luyện

| Yếu tố | Run local thực tế | Paper công bố | Nhận định |
|---|---|---|---|
| Epoch | 50 | 50 | Khớp. |
| Optimizer | SGD, LR 0,002, momentum 0,9, weight decay 0,0001 | SGD, LR đầu 0,002; paper không nêu momentum/weight decay trong phần recipe được công bố | LR/optimizer khớp; 2 tham số còn lại không có giá trị paper để so. |
| LR schedule | MultiStepLR, gamma 0,1 tại epoch 20 và 40 | LR giảm còn 10% mỗi 20 epoch | Khớp theo mô tả. |
| Global batch | 2/GPU × 2 L40 = 4 | Batch 64 trên 8 A800 | **Lệch lớn.** Local có batch nhỏ hơn 16 lần. |
| Optimizer steps | 300 iteration/epoch × 50 = 15.000 steps | Nếu dùng đủ 1.500 video trong một epoch: khoảng 24 steps/epoch × 50 ≈ 1.200 steps | Ước tính từ dataset size và batch paper; local có khoảng 12,5 lần số update, gradient nhiễu hơn, dù số sample exposures toàn quá trình xấp xỉ 60.000 so với 75.000. Không thể suy ra hướng metric chỉ từ số này. |
| Train/validation split | 1.200 train + 300 validation từ 1.500 training videos | Paper nói Nexar có 1.500 clips nhưng không nêu rõ có holdout validation nội bộ hay dùng đủ 1.500 để fit | **Có khả năng** local train ít hơn 20%, nhưng paper chưa công bố split đủ chi tiết để khẳng định tuyệt đối. |
| Chọn checkpoint | `mAUC_full@`; checkpoint được chọn epoch 12 theo validation 300 mẫu | Không công bố validation split hay metric chọn checkpoint | Có thể làm thay đổi checkpoint được benchmark. Không biết epoch 12 có tương đương trọng số paper hay không. |
| Temporal/network setup | SlowOnly-R50, 5 frame/snippet, 30 snippets, ảnh 224×224; lấy mẫu gần 10 Hz | SlowOnly pretrained; 5 frames/snippet, resample 10 fps, 224×224 | Phần lớn khớp ở mức mô tả. Paper không công bố đủ mọi chi tiết sampling để xác nhận `num_clips=30`/cách decode local là tương đương tuyệt đối. |
| Hardware | 2 × L40 | 8 × A800 | GPU khác tự thân không chứng minh gây giảm/tăng chất lượng. Điểm ảnh hưởng trực tiếp hơn là global batch và số optimizer update. |

Nguồn local: [resolved config đã lưu cùng run](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/vis_data/config.py#L145), [config full Nexar](../../configs/riskprop_nexar.py#L5), [schedule SGD](../../configs/_base_/schedules/sgd_50e.py#L6), [run log](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/20261104_085904.log#L1228), và [setup note](../nexar-training-setup.md#L39). Paper công bố recipe tại phần Implementation details và Table 1 trong [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf).

## Lệch trong cách tính auxiliary losses

Paper mô tả BCE anchor ở first/collision frame cho video dương, BCE trên toàn chuỗi cho video âm; negative videos không dùng FFR/AMC. Risk score là sigmoid của logit. Với AMC, paper nói mean risk được tính trên positive videos trong batch. Paper cũng nói collision frames được tăng trọng số so với negative frames, nhưng không chỉ rõ con số.

Trong local checkout:

1. `label_with="constraint"` phù hợp về ý tưởng anchor: positive video có first output = 0, last output = 1; negative video có tất cả nhãn 0 ([config](../../configs/riskprop_nexar_smoke.py#L97), [loss code](../../taa/models.py#L235)).
2. FFR tính `MSE(detach(z[t+1]), z[t])` trên toàn bộ tensor batch, không mask negative samples ([loss code](../../taa/models.py#L249)). Paper nói FFR tắt cho negative videos. Đây là khác biệt về semantics của loss, không phải hyperparameter.
3. AMC chỉ tạo pair loss cho sample dương, nhưng `a_bar` được tính bằng `all_probs.mean()` trên mọi video trong batch ([AMC code](../../taa/models.py#L165)). Paper định nghĩa mean trên positive videos trong batch; các negative samples vì thế ảnh hưởng confidence margin local.
4. Head config đặt `pos_weight=1` ([config](../../configs/riskprop_nexar_smoke.py#L97)), và code BCE không có per-anchor weight riêng ([loss code](../../taa/models.py#L235)). Paper yêu cầu collision anchor có trọng số cao hơn negative anchors, nhưng không công bố trị số. Vì thiếu hệ số trong paper nên không thể tái tạo chính xác chỉ bằng chỉnh một config hiện tại; cần xác minh công thức/trọng số với tác giả hoặc chốt một diễn giải.

Các khác biệt trên cũng có trong [official upstream model source](https://github.com/xingyueye5/RiskProp/blob/main/taa/models.py#L160-L256) ở nhánh đang được đọc; tức chúng có thể là chênh lệch giữa bản cài đặt công khai và mô tả paper, không phải thay đổi chỉ xuất hiện trong local fork. Paper sources: [labeling/training objective and loss](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf). Local `AnticipationHead` lines 165–261 và `loss_by_feat` lines 231–305.

## Yếu tố benchmark, không phải training parameter

- Paper báo cáo Nexar metrics dưới FAR ≤ 0,1 và lấy positive samples tại các horizon trước va chạm; mAUC/mAP là trung bình các horizon 0,5/1,0/1,5 giây. Paper cũng ghi mTTA chỉ tính alarm sau anomaly onset. Các định nghĩa này có trong phần Evaluation metrics của [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf).
- `mAP All=0,895638` của local đến từ Nexar official evaluator và trung bình AP của ba group trên 1.344 ID; đó không tự động đồng nghĩa với AP được giới hạn FAR theo paper.
- Local full mAUC, standardized partial AUC và normalized raw pAUC là ba phép đo được lưu riêng. Paper chỉ công bố `mAUC^0.1`, không nêu đầy đủ normalization để xác định chắc `0,497767` hay `0,735667` là bản so tương đương.
- Local test mTTA chạy theo chuỗi điểm 10 Hz toàn timeline với negative-frame threshold grid; 16 positive bị censor trước onset được báo riêng. Paper định nghĩa mean TTA của true-positive alarms trong vùng FAR ≤ 0,1 và chỉ lấy alarm sau onset. Cần thống nhất miss/censor/threshold aggregation mới so sánh mTTA chuẩn.

Nguồn local protocol: [test report](../../work_dirs/riskprop_nexar_l40_full/benchmark_test_full_mauc_20261104/test_benchmark_report.md), [benchmark JSON](../../work_dirs/riskprop_nexar_l40_full/benchmark_test_full_mauc_20261104/test_results.json), [comparison note](./riskprop-current-run-paper-comparison-20261104.md). Official implementation reference: [RiskProp `AnticipationMetric`](https://github.com/xingyueye5/RiskProp/blob/main/taa/metrics.py).

## Đánh giá nguyên nhân theo mức độ

1. **Chắc chắn tạo khác biệt trong điều kiện học:** global batch 4 thay vì 64 và 15.000 so với khoảng 1.200 optimizer steps. Không thể dự đoán delta metric sẽ theo hướng nào chỉ từ điều này.
2. **Có thể làm giảm dữ liệu học/đổi checkpoint:** holdout 300/1.500 video và chọn epoch 12 bằng validation full mAUC; paper không nói rõ hai lựa chọn tương ứng.
3. **Khác biệt định nghĩa benchmark làm số nhìn lệch:** đặc biệt mAP và partial mAUC; trước khi dùng làm bằng chứng model tốt hơn/kém hơn, cần cùng input IDs, các horizon, FAR rule, formula và xử lý miss.
4. **Khả năng khác biệt algorithm so với mô tả paper:** negative-video FFR, AMC `a_bar` không positive-only và không có higher weight cho collision anchor. Đây là điểm cần audit/fix trước một run được gọi là reproduction paper-exact.
5. **GPU model L40 vs A800:** không đủ căn cứ để quy chênh lệch chất lượng cho tên GPU; batch, update schedule, dataset split, seed/nondeterminism và loss/evaluator semantics là confound cụ thể hơn.

## Khuyến nghị để kiểm tra nguyên nhân có căn cứ

Không nên đổi hàng loạt LR/epoch trước. Để xác định nguồn chênh lệch theo từng phần:

1. Chốt evaluator paper-compatible trên cùng Nexar 1.344 test IDs: positive horizons, negative sampling, FAR≤0,1 cho AP/AUC, công thức partial AUC, và định nghĩa mTTA/miss sau onset. Nếu predictions cũ đủ để tính cùng protocol thì tính lại trước, không cần huấn luyện lại.
2. Xác minh với paper/repository những phần loss chưa khớp (negative FFR, positive-only `a_bar`, collision anchor weight). Sửa thành một biến thể paper-consistent có kiểm thử đơn vị trước khi retrain.
3. Khi đủ tài nguyên, chạy ablation có kiểm soát: cùng code, seed, data split và evaluator; lần lượt đổi global batch (4 so với 64/gradient accumulation), số train IDs (1.200 so với 1.500 nếu protocol cho phép), rồi checkpoint-selection rule. Ghi lại validation và test riêng.

Đây là kế hoạch chẩn đoán, chưa chạy thêm benchmark/training trong audit này.
# Metric naming update (2026-11-10)

This audit records metrics from the earlier implementation, which reported
standardized and normalized partial AUC separately. Current RiskProp code
reports only normalized partial AUC as `mAUC_fpr_0.1@` for validation and
`mAUC_fpr_0.1` for Nexar test evaluation; the older values below remain a
historical comparison and were not recomputed under the current report schema.

