# RiskProp, TOP và Nexar: đối chiếu protocol đánh giá

Ngày rà soát: 09/11/2026. Phạm vi: paper RiskProp, paper TOP, paper/dataset Nexar, mã nguồn TOP chính thức, evaluator Nexar được phát hành cùng dataset, và code RiskProp hiện có trong workspace. Ghi chú được cập nhật sau thay đổi mTTA cùng ngày; không chạy lại benchmark.

## Kết luận ngắn

1. **Nexar official benchmark** nhận một score cho mỗi video đã được cắt theo TTE 500/1000/1500 ms; evaluator tính AP riêng từng group rồi lấy trung bình. Đây là AP đầy đủ trên PR curve, không giới hạn FAR. [Nexar paper §4.1–4.2](https://arxiv.org/html/2503.03848), [official evaluator](../../../datasets/Nexar/evaluate_submission.py#L7-L26).
2. **TOP paper** định nghĩa AUC dưới FAR ≤ 0.1 (paper nói FPR tương đương FAR), và mTTA/recall dưới cùng ràng buộc. Nó đánh giá positive clips tại ba khoảng 0.5/1.0/1.5 giây trước tai nạn so với negative segments dài 0.5 giây. TOP paper không định nghĩa hay báo mAP trong phần metric/results. [TOP paper §4](https://arxiv.org/html/2510.22260#S4).
3. **TOP code chính thức** lại có mAP, nhưng tính bằng `average_precision_score` trên toàn PR curve; AUC giới hạn FPR được cài là diện tích ROC thô chia cho ngưỡng FPR, không phải sklearn standardized pAUC. README mô tả mAP ở ba horizon và mAUC ở FPR ≤ 10%. [TOP metric code](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/metrics.py), [TOP AUC/AP code](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/evaluate.py), [TOP README](https://github.com/happytianhao/TOP#evaluation-metrics).
4. **RiskProp paper** mô tả causal sliding window trên phần trước va chạm, snippet 5 frame, 10 fps, resize 224; các metric mAP/mAUC tổng hợp ba horizon 0.5/1.0/1.5 giây. Paper nói mọi metric có FAR ≤ 0.1, nhưng không cung cấp đầy đủ code-level định nghĩa AP có ràng buộc này. [RiskProp §4.1](https://arxiv.org/html/2603.27165#S4.SS1).
5. **RiskProp local** có hai luồng. Luồng mAP cơ bản dùng `SampleFramesBeforeAccident`; dữ liệu test bỏ `accident_frame` nhưng giữ `target`, nên transform lấy snippet ở đuôi positive và đầu negative. Luồng mTTA riêng dùng `SampleFullVideoCausalFrames`, không chọn frame theo target, và được gọi tự động bởi `tools/dist_test.sh` khi chạy config `riskprop_nexar.py`. Mã nguồn: [dataset annotations](../../taa/datasets.py#L304-L365), [cơ bản transform](../../taa/transforms.py#L43-L106), [full-video transform](../../taa/transforms.py#L109-L144), [mTTA config](../../configs/riskprop_nexar_mtta.py#L13-L40), [test wrapper](../../tools/dist_test.sh#L28-L81).

## 1. Nexar: test crops, metric và evaluator

### Protocol được công bố

- Test set có 1.344 clip khoảng 10 giây, được tạo từ 568 video nguồn. Mỗi positive có thể có tối đa ba crop kết thúc tại `t_event − TTE`, với TTE 500/1000/1500 ms; crop ở TTE xa hơn chỉ tạo nếu thời gian event-alert đủ dài. Với negative, tác giả tạo event/alert times giả để khớp phân bố positive. Public/private là hai nửa test. [Nexar paper §4.1](https://arxiv.org/html/2503.03848#S4.SS1).
- Model nộp một confidence score cho mỗi test clip. Evaluator nhóm score theo TTE 500/1000/1500, tính AP từng nhóm bằng công thức weighted precision–recall area (scikit-learn), rồi lấy trung bình ba AP. [Nexar paper §4.2](https://arxiv.org/html/2503.03848#S4.SS2).
- Evaluator chính thức không giải mã video hay chọn frame. Nó merge prediction với solution theo ID, gọi `average_precision_score` trong từng `group`, trung bình group AP và báo riêng Public/Private. Không có bước lọc threshold FAR trong script. [Nexar evaluator](../../../datasets/Nexar/evaluate_submission.py#L7-L26); [dataset README](../../../datasets/Nexar/README.md#L55-L63).

### Hệ quả

Việc chọn crop theo mốc TTE xảy ra khi **tạo test dataset**; tại inference, benchmark đưa clip đã cắt cho model và nhận một score. Dùng event metadata để định nghĩa các sample/horizon theo protocol không đồng nghĩa model được cấp target label. Nhưng model-side sampling theo `target` phải khớp các crop/horizon đã định trước; benchmark evaluator tự nó không quy định head/tail frame extraction.

## 2. TOP: paper so với mã nguồn chính thức

### TOP paper

- Paper định nghĩa `FPR = FAR`; AUC giới hạn là diện tích ROC đến điểm FPR ≤ λ, λ mặc định 0.1. Nó lấy positive clips từ 0.5–1.0, 1.0–1.5, 1.5–2.0 giây trước tai nạn, và mỗi horizon có số negative segment an toàn dài 0.5 giây tương đương; mAUC là trung bình ba AUC này. [TOP §4](https://arxiv.org/html/2510.22260#S4).
- TOP paper mô tả TTA chỉ tính sau anomaly onset; alarm trước onset được xem là false alarm, missed accident có TTA=0. FAR constraint áp cho mTTA. [TOP §4](https://arxiv.org/html/2510.22260#S4).
- **Khác với mô tả paper**, mã TOP hiện công khai tính `tta_old` trên `pred[:accident_ind+1]`, tức tìm cảnh báo từ đầu chuỗi đến frame tai nạn, không dùng `abnormal_start_ind` làm điểm bắt đầu. Nếu frame tai nạn không có trong chuỗi, evaluator đặt `accident_ind=0`. RiskProp local hiện khớp quy tắc mã này. [TOP metric implementation](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/metrics.py).
- TOP paper nói test dùng sliding window stride cố định trên **toàn bộ các frame có sẵn**, gồm trước, trong và sau tai nạn. Metric section nói AP là metric phổ biến trong các công trình trước, nhưng TOP paper không định nghĩa công thức AP giới hạn FAR và bảng kết quả chính tập trung vào AUC/mAUC, recall và mTTA. [TOP §3.2](https://arxiv.org/html/2510.22260#S3.SS2), [TOP §4–5](https://arxiv.org/html/2510.22260#S4).

### TOP repo

- Config Nexar kế thừa `top_cap.py`, với `clip_len=5`, `num_clips=30`; test loader trỏ tới annotation holdout `nexar_val_annotations.csv`, không phải hidden official Kaggle test. [TOP Nexar config](https://github.com/happytianhao/TOP/blob/main/configs/accident_anticipation/top/top_nexar.py), [base config](https://github.com/happytianhao/TOP/blob/main/configs/accident_anticipation/top/top_cap.py).
- `SampleFramesBeforeAccident` dùng event frame để căn sample dương vào tai nạn. Negative không có accident frame lấy từ đầu video nếu video đủ dài; nếu có event frame thì lấy vùng an toàn trước event. Test/validation dùng sample cố định, còn train positive/negative có jitter ngẫu nhiên. [TOP transform](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/transforms.py).
- Metric code lấy score ở các cửa sổ 0.5/1.0/1.5 giây trên chuỗi dự đoán positive và so với cửa sổ cuối 0.5 giây negative. AP gọi `average_precision_score` trên toàn bộ samples, không cắt ở FAR=0.1. Custom AUC cộng trapezoid đến FPR limit rồi chia cho limit; code không nội suy biên như sklearn standardized pAUC. [TOP metrics](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/metrics.py), [TOP evaluator functions](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/evaluate.py).
- TOP README công bố mAP là trung bình AP@0.5/1.0/1.5; mAUC ở FPR≤10%; implementation code cho thấy mAP vẫn là full AP. Đây là khác biệt giữa cách ghi tóm tắt README/paper và phần code AP. [TOP README](https://github.com/happytianhao/TOP#evaluation-metrics), [TOP AP implementation](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/metrics.py).
- TOP repo hiện công khai có thể đã phát triển sau bản NeurIPS: RiskProp repo được fork từ TOP và README TOP hiện liệt kê RiskProp follow-up. Vì vậy các chi tiết repo ở đây là **implementation hiện tại**, không thể mặc định là artifact chính xác dùng để tạo paper TOP năm 2025. [TOP repo](https://github.com/happytianhao/TOP), [RiskProp fork](https://github.com/xingyueye5/RiskProp).

## 3. RiskProp: paper và local code

### RiskProp paper

- §4.1 nói theo preprocessing Nexar; test có 1.344 clip ~10s; lấy positive ở bốn khoảng trước event `[0,0.5)`, `[0.5,1)`, `[1,1.5)`, `[1.5,2)` giây và negative safe clips cân bằng. AUC/AP tính khi FAR≤0.1; mAUC/mAP trung bình các horizon 0.5/1.0/1.5s, không gồm 0.0s trong mean. [RiskProp §4.1](https://arxiv.org/html/2603.27165#S4.SS1).
- Phần implementation details nói test causal sliding window trên **pre-collision portion** của video tai nạn; resize 224×224, lấy mẫu 10fps, 5 frame/snippet. Paper không đưa chính xác code cho frame-index boundary, cách lấy từng negative safe crop, hoặc AP/FAR tie handling. [RiskProp §4.1, implementation details](https://arxiv.org/html/2603.27165#S4.SS1).

### Code RiskProp local hiện tại

- Full Nexar config kế thừa smoke config; validation và lượt test mAP dùng 30 snippet, mỗi snippet 5 frame, với `SampleFramesBeforeAccident`. Lượt mTTA dùng config riêng `riskprop_nexar_mtta.py`, lấy toàn timeline bằng `SampleFullVideoCausalFrames`, batch size 1 và xuất timestamps. Wrapper `tools/dist_test.sh` chạy cả hai lượt khi nhận `riskprop_nexar.py`. [configs](../../configs/riskprop_nexar.py#L1-L29), [base sampling config](../../configs/riskprop_nexar_smoke.py#L7-L37), [mTTA config](../../configs/riskprop_nexar_mtta.py#L13-L40), [wrapper](../../tools/dist_test.sh#L28-L81).
- Với Nexar test rows, `accident_frame` và `abnormal_start_frame` được đặt `None`, còn `target` vẫn có. `SampleFramesBeforeAccident` dùng target: positive được lấy ở đuôi clip, negative ở đầu clip. Với `num_clips=30, clip_len=5`, đây là chuỗi snippet cố định, không phải trượt cửa sổ trên toàn bộ clip 10 giây Nexar. Evaluator chỉ ghi `pred[-1]` thành một score/video. [dataset construction](../../taa/datasets.py#L304-L365), [transform](../../taa/transforms.py#L43-L106), [score export](../../taa/metrics.py#L114-L121), [sampling config](../../configs/riskprop_nexar_smoke.py#L7-L10).
- Local Nexar mAP evaluator dùng official group AP, không FAR-limited; mAUC full và hai partial variants được báo riêng. Local validation mAP cũng là full AP. [local metrics](../../taa/metrics.py#L270-L312), [submission evaluator](../../tools/analysis_tools/evaluate_nexar_submission.py#L24-L46).

## 4. Bảng so sánh

| Hạng mục | Nexar paper/evaluator | TOP paper/repo | RiskProp paper | RiskProp local |
|---|---|---|---|---|
| Tập test/crop | Clip đã pre-crop kết thúc 0.5/1.0/1.5s trước event; negative crop được tạo để cân bằng | Paper nói sliding window stride cố định; repo dùng event-anchored positive / safe-negative sampling cho holdout | Causal sliding trên phần pre-collision; đánh giá crop theo 4 interval | 30×5 snippet trong vùng do target chọn; chỉ xuất score cuối |
| Score đầu vào evaluator | 1 score/clip | Theo frame/horizon; repo cũng ghi score cuối snippet cho submission | Score theo horizon/frame rồi chấm ở từng τ | mAP: 1 score/clip; mTTA: chuỗi score theo thời gian từ full-video config |
| AP/mAP | AP đầy đủ từng group; mean 3 TTE groups; không FAR | Repo AP đầy đủ; paper không đặc tả/báo AP constrained | Paper nói AP dưới FAR≤0.1 nhưng không đủ chi tiết implementation | mAP official group AP full PR, không FAR constraint |
| AUC | Không phải metric official Nexar challenge | Paper trapezoid partial AUC đến FPR≤0.1; repo chia raw area cho limit | mAUC dưới FAR≤0.1 trên 0.5/1.0/1.5 | Full AUC + sklearn standardized pAUC + normalized raw pAUC riêng |
| TTA | Không thuộc Kaggle score chính | Paper: sau onset, FAR≤0.1, miss=0; repo code: tìm alert từ đầu chuỗi đến event | Paper: chỉ alert sau onset, FAR≤0.1 | Một mTTA theo TOP repo code: tìm từ đầu chuỗi đến event; event ngoài clip ánh xạ index 0; miss đóng góp 0 |

## 5. Đánh giá mức độ khớp và đề xuất

### Đã khớp / tương đồng

- Local mAP đánh giá đúng toàn bộ 1.344 test ID theo ba Nexar groups và trung bình group AP như official evaluator; đây là phép đo phù hợp khi gọi là **Nexar official grouped mAP**.
- Local dùng snippet 5 frame ở 10Hz và score theo horizon trong validation; đây là lựa chọn có cơ sở từ mô tả paper TOP/RiskProp.
- Local có evaluator báo full AUC, standardized partial AUC và normalized raw pAUC riêng. Luồng benchmark `tools/dist_test.sh riskprop_nexar.py ...` chạy thêm config full-video để xuất temporal predictions và test mTTA; chạy trực tiếp `tools/test.py` với config mAP cơ bản thì không tự chạy lượt thứ hai này.

### Chưa khớp / chưa đủ bằng chứng

1. **Sampling test:** local target-dependent head/tail selection không phải thao tác được Nexar evaluator yêu cầu. Nexar đã cung cấp clip tại TTE; nên dùng nhất quán clip đó và tính score tại điểm quan sát cuối/đúng horizon, không để target chọn đầu hay cuối. Offline event metadata dùng để xác định group/crop lúc chuẩn bị dataset là hợp lệ nếu theo đúng protocol; vấn đề là quy tắc sampling hiện tại không chứng minh tương đương crop đó.
2. **Horizon outputs:** local submission CSV chỉ có một score cuối/clip. Cần xác minh Nexar group trên 1.344 clip có đại diện trực tiếp TTE; nếu có, score cuối mỗi clip có thể khớp official benchmark. Để tái hiện riêng RiskProp mAP theo 0.5/1.0/1.5 cần đảm bảo từng prediction gắn đúng group/horizon và đúng negative pairing.
3. **AP và FAR:** Nexar official mAP full AP không ràng buộc FAR; RiskProp paper tuyên bố AP với FAR≤0.1; TOP paper/repo không cung cấp một implementation AP-limited tương đương (TOP paper không báo AP, TOP repo tính full AP). Không nên gọi Nexar `mAP All` local là paper-exact mAP RiskProp cho tới khi xác minh/định nghĩa được constrained AP.
4. **Partial AUC:** TOP repo chia raw partial area cho 0.1; paper AUC equation chỉ mô tả diện tích tới FAR limit; RiskProp paper không ghi công thức chuẩn hóa đủ rõ. Local standardized và normalized scores đều có ích, nhưng chưa thể chốt score nào tái hiện con số paper nếu thiếu implementation gốc.

### Bước tiếp theo được đề xuất

1. Dùng metadata Nexar để xác minh từng test video/group là TTE 0.5/1.0/1.5 và giữ nguyên clip đã phát hành.
2. Trong inference test, cố định điểm dự đoán tại cuối clip cho mọi mẫu; chỉ sau inference mới join target/group từ solution.
3. Xuất AP theo từng TTE group, official Nexar mAP, và các metric paper-inspired riêng; ghi rõ AP full hay FAR-constrained.
4. Nếu cần tuyên bố paper-exact, tìm implementation evaluator/công thức gốc của RiskProp; không suy ra AP constraint chỉ từ TOP repo vì paper và code TOP không cùng đặc tả AP.

# Metric naming update (2026-11-10)

This protocol audit describes the prior implementation, which reported both
standardized and normalized partial AUC. Current RiskProp output keeps only
the normalized formula, named `mAUC_fpr_0.1@` in validation and
`mAUC_fpr_0.1` in test evaluation. Historical score descriptions below refer
to the earlier code.

