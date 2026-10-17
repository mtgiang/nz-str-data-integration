# Bản sửa bài cho MDPI *Data*: sửa gì, vì sao, và sửa ở đâu

Nguyên tắc chung: **không chạy lại và không thay đổi mô hình frozen hay kết quả T10.** Mọi con số mới đều được tính từ các prediction và file đã lưu, bằng `revision_addons.py`, rồi tự động ghi vào `revision_macros.tex`. Trong manuscript không có con số nào được gõ tay.

---

## A. Các điểm yếu và cách khắc phục

| # | Điểm yếu (góc nhìn reviewer) | Sửa thế nào | Vì sao | Vị trí trong `main.tex` |
|---|---|---|---|---|
| 1 | Bài cho tạp chí *dữ liệu* nhưng không công bố dữ liệu | Thêm §3.9 *Shareable Data Products*. Script R5 tạo gói dữ liệu chia sẻ được (mục B). Viết lại Data Availability | *Data* đánh giá cao tính tái sử dụng; thiếu dữ liệu là lý do bị yêu cầu sửa lớn | §3.9 (mới), Data Availability, Supplementary (Table S6) |
| 2 | Mô hình chính vẫn chứa 2 biến FAIL | §4.3 nêu rõ: trong cùng một bản cài đặt, bản 51 biến có PR-AUC dev cao hơn (0.0792 so với 0.0770), nên quy tắc chọn đã định trước cũng sẽ chọn nó → **khuyến nghị dùng bản 51 biến**; vẫn báo cáo bản 53 biến để trung thành với protocol | Trả lời trước câu hỏi "sao không dùng bản sạch?" mà không mở lại T10 | §4.3, Abstract, Conclusions |
| 3 | PR-AUC dev 0.0749 (frozen) khác 0.0770 (refit) | Nói thẳng: bản cài đặt lại không tái lập chính xác giá trị frozen, nên chỉ so sánh các đặc tả **trong cùng một bản cài đặt**. Không bịa nguyên nhân | Trung thực; reviewer sẽ thấy ngay chênh lệch này | §4.3 |
| 4 | Ablation không có khoảng tin cậy | **R1**: so sánh cặp có/không từng family, đếm số cửa sổ (trên 6) mà family giúp, kèm 95% CI bằng TA-cluster bootstrap ghép cặp (cùng một lần rút TA cho cả hai đặc tả và mọi cửa sổ) → Bảng `tab:family_paired` | Hiệu ứng ablation chỉ khoảng 0.001, nên không có CI thì không thể kết luận | §3.7 (phương pháp), §4.5 + bảng mới |
| 5 | Nhãn có nhiễu (12.76% quay lại, T3 là 25%) | **R2**: chấm lại prediction dev A5 với nhãn "biến mất xác nhận" (vắng ở t+1 và t+2, làm được cho T4–T9), so sánh với nhãn gốc, và tính rủi ro dự đoán trung bình của nhóm quay lại so với nhóm biến mất thật → Bảng `tab:confirmed_label` | Cho thấy mô hình có bắt được việc rời nền tảng thật hay chỉ bắt nhiễu thu thập dữ liệu | §4.7 + bảng mới |
| 6 | `month_sin/cos` đứng đầu SHAP | Ghi rõ tháng 7 (T10) không có trong các tháng huấn luyện (tháng 10 đến tháng 6), nên mô hình ngoại suy; ảnh hưởng đã bị giới hạn bởi phân tích ở §4.3 | Tránh reviewer cho rằng mô hình "học được mùa vụ" | §4.8 |
| 7 | `open_ratio_w4` có thể gần như hiển nhiên | Diễn giải là host đóng lịch trước khi rời đi: thông tin hợp lệ tại origin, nhưng là biểu hiện sớm của quyết định của chính host | Chủ động thừa nhận thay vì để reviewer phát hiện | §4.8 |
| 8 | "Independent audit" trong khi audit do AI thực hiện | Đổi thành **post-hoc audit** ở cả 5 chỗ | AI audit không "độc lập" theo nghĩa của reviewer; giữ nguyên khai báo GenAI | toàn bài |
| 9 | Dấu nháy sai ở chú thích Bảng 3 | ‘Scraped late” → “Scraped late” | Lỗi trình bày | Bảng 3 |
| 10 | Abstract quá 200 từ nếu thêm ý mới | Viết lại abstract khoảng 200 từ, có câu về construction và bản 51 biến | MDPI giới hạn abstract khoảng 200 từ | Abstract |

## B. Dữ liệu chia sẻ: vì sao làm như vậy

**Chính sách của Inside Airbnb.** Trang *Data Policies* yêu cầu **"Do not republish the data"**, dù trang tải dữ liệu ghi giấy phép CC BY 4.0. Bản cũ chỉ ghi CC BY 4.0. Mình chọn cách hiểu thận trọng: **không công bố bất kỳ bản ghi cấp listing nào**, kể cả khi đã ẩn danh.

Gói `10_revision_out/shareable_data/` do R5 tạo ra gồm:

| File | Nội dung | Vì sao an toàn và vẫn hữu ích |
|---|---|---|
| `ta_transition_aggregates.csv` | Số listing, tỷ lệ biến mất thực tế, rủi ro dự đoán trung bình theo **TA × transition**; ô có dưới 10 listing bị ẩn | Là số tổng hợp, không phải dữ liệu Airbnb gốc; vẫn kiểm tra được calibration theo khu vực |
| `ta_transition_context_layer.csv` | Biến construction và rental đã căn theo ngày phát hành, đúng như mô hình dùng | Nguồn Stats NZ/MBIE theo CC BY nên được phép chia sẻ. **Đây là "dấu ấn construction" có thể tái sử dụng** |
| `dev_metrics_by_window.csv`, `R1–R4*.csv` | Metric và ablation có CI | Kiểm chứng được các bảng trong bài |
| `SYNTHETIC_listing_origin_sample.csv` | 5,000 dòng **giả**, đúng schema; mỗi cột được rút ngẫu nhiên độc lập nên không dòng nào là listing thật | Người khác **chạy được code** mà không cần dữ liệu thật |
| `DATA_README.md`, `CHECKSUMS.csv` | Mô tả, giấy phép, SHA-256 | Đáp ứng tiêu chí FAIR |

Ngoài ra, `SNAPSHOT_MANIFEST.csv` (đã có từ bản audit) kèm code đóng vai trò **bộ dựng lại**: ai lấy cùng các snapshot từ Inside Airbnb (qua archived-data request) có thể kiểm tra SHA-256 rồi tái tạo toàn bộ kết quả cấp listing.

Script có **bước kiểm tra tự động**: nếu bất kỳ file nào chứa cột id, tên, toạ độ, URL hay text, script dừng lại với lỗi `POLICY CHECK FAILED`.

> Khuyến nghị thêm: gửi một email ngắn tới data@insideairbnb.com mô tả gói này (chỉ số tổng hợp, không có bản ghi) để có xác nhận bằng văn bản. Đây không phải điều kiện bắt buộc để nộp bài.

## C. Construction: giữ "dấu ấn" mà không nói quá

**Thực tế từ dữ liệu:** ở độ phân giải TA và theo tháng, consents gần như không cải thiện khả năng phân biệt cho dự báo một tháng ở cấp listing. Không thể và không nên "làm cho" construction trở nên quan trọng.

**Cách giữ dấu ấn một cách trung thực, theo 4 lớp:**

1. **Construction là một thành phần phương pháp, không chỉ là một biến.** Lớp construction được căn theo **ngày phát hành chính thức** (khớp 13/13 với Stats NZ). §3.4 và đóng góp thứ nhất trong Introduction nói rõ điều này. Đây là kỹ thuật dùng lại được cho mọi chương có dữ liệu consents.
2. **Kiểm tra *ở đâu* construction có ý nghĩa** (§4.6 mới, R3): so sánh mức cải thiện theo tercile cường độ consents của TA. Nếu ở TA nhiều xây dựng mà mức cải thiện cao hơn, đó là một phát hiện có giá trị; nếu không, đó vẫn là bằng chứng chặt chẽ.
3. **Đổi đơn vị phân tích sang cấp khu vực** (R4): tương quan giữa bối cảnh consents và tỷ lệ biến mất của TA, trong từng transition. Construction là tín hiệu cấp khu vực nên có thể liên quan đến **biến động thị trường của khu vực** hơn là rủi ro của từng listing. Bài chỉ diễn đạt là "mô tả liên hệ", không phải nhân quả.
4. **Thảo luận định hướng** (§5.3 mới, *The Role of Construction Context*): giải thích vì sao tín hiệu yếu (biến thiên giữa khu vực, chậm, mất 1–2 năm từ consent đến khi nhà hoàn thành) và chỉ ra bước tiếp theo: độ phân giải nhỏ hơn (SA2 hoặc từng consent), horizon dài hơn, outcome cấp khu vực.

**Liên kết với thesis (viết trong thesis, không phải trong bài):**
- Chương này = *lớp tích hợp dữ liệu xây dựng có kiểm soát thời điểm* cùng một kiểm định nghiêm ngặt.
- Kết luận "TA-level consents không đủ cho sàng lọc listing ngắn hạn" là **động lực trực tiếp** cho chương *housing production monitoring* (dữ liệu xây dựng chi tiết hơn) và cho các phân tích ở cấp khu vực.
- `ta_transition_context_layer.csv` là **sản phẩm dữ liệu dùng chung** giữa các chương.
- Câu chuyện chung của thesis: *AI trong xây dựng và bất động sản cần (i) căn đúng thời điểm thông tin, (ii) đúng độ phân giải không gian, (iii) kiểm định có thể cho kết quả âm.*

**Tiêu đề (tác giả quyết định).** Phương án thay thế được ghi dạng comment ở đầu `main.tex`:
*"Integrating Platform, Building-Consent and Rental-Market Data for Short-Term-Rental Screening: A Point-in-Time Audited Temporal Evaluation from New Zealand"*. Phương án này đưa construction vào tiêu đề mà không hứa hẹn hiệu ứng.

## D. Câu chữ phụ thuộc vào kết quả thật

Các câu ở §4.5, §4.6 và §4.7 được viết **trung tính**, ví dụ: "khoảng tin cậy chứa 0 nghĩa là không phân biệt được với biến thiên do lấy mẫu lại TA". Sau khi chạy script, **đọc lại 4 đoạn có macro** và chỉnh câu kết luận cho đúng với số thật:
- Nếu CI của construction **chứa 0**: giữ nguyên câu.
- Nếu tercile cao có mức cải thiện **dương rõ ràng** trong khi tercile thấp không có: thêm một câu nêu rõ điều đó ở §4.6 và §5.3.
- Nếu nhãn "xác nhận" cho ROC-AUC **cao hơn** nhãn gốc: thêm một câu rằng mô hình tách được việc rời nền tảng thật khỏi việc biến mất tạm thời.

## E. Các file trong `10_revision/`

| File | Dùng để |
|---|---|
| `revision_addons.py` | Script R1–R5 cùng phần sinh LaTeX |
| `revision_addons.ipynb` | Chạy từng bước trong Jupyter |
| `manuscript/main.tex` | Bản đã sửa (bản gốc lưu ở `main_before_revision.tex`) |
| `MAIN_REVISION.diff` | Diff từng dòng giữa bản gốc và bản sửa |
| `REVISION_RATIONALE.md` | Tài liệu này |

## F. Bản cuối và cách compile

- `manuscript/main.tex` **compile được ngay**: 21 trang, không lỗi, chỉ chứa các số đã kiểm chứng. Bản xem trước: `manuscript/main_preview.pdf`.
- Khi có file `revision_macros.tex` và 3 file `tab_*.tex` (do `revision_addons.py` sinh ra) nằm cạnh `main.tex`, LaTeX **tự động** thêm phần R1–R4: bảng ablation có CI, §4.6 về construction, robustness nhãn xác nhận (khoảng 22 trang). Không cần sửa tay.
- `cover_letter.md`: bản nháp thư gửi editor; cần điền reviewer đề xuất.
- Còn 9 marker `[AUTHOR ACTION REQUIRED]` (khai báo tác giả và DOI). Chỉ tác giả mới điền được.
