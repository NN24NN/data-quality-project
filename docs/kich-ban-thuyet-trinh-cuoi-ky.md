# Kịch bản thuyết trình cuối kỳ (18 slide, khoảng 22 phút)

Slide: https://claude.ai/artifact/HjqCwSnmjw6MeTtBAB23DR (bản riêng tư của người dùng).

**Cách dùng:** lời thoại viết như văn nói, đọc thẳng hoặc diễn đạt lại bằng lời của mình. Số liệu lấy từ báo cáo đã chốt (`docs/ket-qua-thuc-nghiem.md`). **Phân công người nói chỉ là đề xuất** theo phân công trong báo cáo (Nam: giai đoạn 0–5, Mạnh: giai đoạn 6–9); đổi tùy ý.

| Phần | Slide | Người nói | Thời lượng |
|---|---|---|---|
| Nhắc lại giữa kỳ | 1–4 | Nam | 3,5 phút |
| Từ giữa kỳ đến nay | 5–8 | Nam (5–6), Mạnh (7–8) | 5 phút |
| Kết quả RQ1–RQ4 | 9–15 | Mạnh | 10,5 phút |
| Thảo luận và kết luận | 16–18 | Mạnh (16–17), Nam (18) | 3,5 phút |

---

## Slide 1 — Trang bìa (0,5 phút, Nam)

“Em xin chào thầy cô và các bạn. Nhóm em là Nguyễn Hoàng Nam và Vũ Đức Mạnh. Hôm nay nhóm em báo cáo cuối kỳ đề tài *Khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết tác vụ, dự đoán mức suy giảm hiệu năng mô hình downstream*. Ở giữa kỳ nhóm đã trình bày phần thiết kế và chuẩn bị dữ liệu, nên hôm nay em nhắc nhanh phần đó và tập trung vào những gì nhóm làm được từ đó đến nay.”

*Lưu ý: ở giữa kỳ tên đề tài là “có nhận biết ngữ cảnh, gắn với hiệu năng…”. Nói một câu rằng tên đã đổi cho chính xác hơn: “ngữ cảnh” trong đề tài chính là loại tác vụ.*

## Slide 2 — Bài toán và khoảng trống (1 phút, Nam)

“Bài toán của nhóm là: nhìn vào chất lượng của một bộ dữ liệu, ta có dự đoán được mô hình sẽ mất bao nhiêu hiệu năng không. Hai công trình nền tảng đều chưa trả lời câu hỏi này. Bài báo gốc chứng minh ảnh hưởng của chất lượng dữ liệu phụ thuộc vào tác vụ và thuật toán, nhưng chỉ vẽ đường cong, không có mô hình dự đoán. DQSOps có mô hình dự đoán, nhưng nhãn là một điểm PCA tự định nghĩa, một điểm chung cho mọi tác vụ, không gắn với hiệu năng thật. Đề tài của nhóm lấp khoảng trống này: nhãn là mức suy giảm hiệu năng downstream thật, mô hình nhận biết tác vụ, giải thích bằng SHAP, và profiling mở rộng được bằng Spark.”

## Slide 3 — Bốn câu hỏi nghiên cứu (1 phút, Nam)

“Nhóm trả lời bốn câu hỏi. Một: ảnh hưởng của từng chiều chất lượng có thật sự phụ thuộc vào tác vụ và thuật toán không. Hai: một meta-model học từ mức suy giảm thật có hơn các điểm chất lượng cố định không. Ba: mức quan trọng của các chiều có khác nhau theo tác vụ, và khác trọng số cố định của PCA không. Bốn: bước profiling có mở rộng được khi dữ liệu vượt bộ nhớ không. Phần kết quả sau đây sẽ trả lời lần lượt từng câu.”

## Slide 4 — Thiết kế thí nghiệm (1 phút, Nam)

“Về thiết kế, nhóm dùng ba bộ dữ liệu: Beijing cho regression, HIGGS cho classification, Covertype cho clustering. Mỗi lượt làm bẩn một trong năm chiều chất lượng ở bốn mức, từ 0 đến 0,8, và làm bẩn cả train lẫn test vì đó là tình huống thực tế nhất. Có sáu thuật toán và ba baseline để so sánh. Tổng cộng 312 lượt, thu gọn từ 4.905 lượt của bài gốc vì giới hạn tài nguyên. Profiling ở đây là bước tóm một bộ dữ liệu thành năm điểm chất lượng trong khoảng 0 đến 1; đó chính là đầu vào của meta-model.”

## Slide 5 — Hoàn thành 10 trên 10 giai đoạn (1 phút, Nam)

“Ở giữa kỳ nhóm mới xong giai đoạn 0 đến 4, tức là thiết kế, dữ liệu, profiling, làm bẩn và các mô hình downstream. Từ đó đến nay nhóm hoàn thành năm giai đoạn còn lại: chạy Experiment Runner, huấn luyện meta-model, đánh giá, thí nghiệm Spark, và biểu đồ cùng báo cáo. Ba con số tóm tắt: 312 trên 312 lượt thí nghiệm đã chạy xong, số test tăng từ 93 lên 134, và Spark đã profiling 22 triệu dòng thay vì 11 triệu như dự kiến.”

## Slide 6 — Experiment Runner (1 phút, Nam)

“Experiment Runner chạy xong 312 lượt trên Google Colab trong 414,5 phút, tức khoảng 6,9 giờ, nhanh hơn ước tính khoảng 8 giờ ở giữa kỳ. Regression 96 lượt mất 140 phút, classification 120 lượt mất 148 phút, clustering 96 lượt mất 126 phút. Một chi tiết kỹ thuật: ở mức bẩn bằng 0, mỗi cặp thuật toán và seed chỉ huấn luyện thật một lần rồi chép sang các chiều khác, nên có 252 trong 312 dòng được huấn luyện thật. Nhóm kiểm tra bảng kết quả: đủ 312 dòng, không trùng khóa, và chênh lệch lớn nhất ở mức bẩn 0 chỉ cỡ 10 mũ trừ 16, đúng như kỳ vọng vì cùng dữ liệu, cùng seed.”

## Slide 7 — Nhãn Δ và meta-model (1,5 phút, Mạnh)

“Em xin trình bày phần học mô hình. Nhãn của meta-model là Δ bằng hiệu năng trên dữ liệu sạch trừ hiệu năng trên dữ liệu bẩn, cùng thuật toán và cùng seed. Δ lớn nghĩa là hiệu năng giảm nhiều. Nhóm ghép baseline theo cùng seed để loại phần dao động do seed, thứ mà mô hình không thể học vì không nhìn thấy seed. Meta-model là Gradient Boosting với 14 đặc trưng: năm điểm profile, ba biến one-hot cho tác vụ và sáu biến one-hot cho thuật toán. So với kế hoạch giữa kỳ là 10 đặc trưng, nhóm đổi họ thuật toán thành one-hot sáu thuật toán, vì k-Means và GMM không thuộc họ tuyến tính hay cây. Mô hình có 100 cây, độ sâu 3, không tinh chỉnh, vì chỉ có 312 dòng nên tinh chỉnh trên chính tập đánh giá sẽ làm kết quả lạc quan giả tạo. Cuối cùng SHAP cho trọng số của từng chiều theo từng tác vụ.”

## Slide 8 — Cách đánh giá (1,5 phút, Mạnh)

“Cách đánh giá có một thay đổi quan trọng so với giữa kỳ. Kế hoạch ban đầu là chia ngẫu nhiên, nhưng khi có bảng 312 dòng nhóm nhận ra sẽ bị rò rỉ: ba seed của cùng một cấu hình gần như là ba bản sao, nên mô hình có thể nhớ đáp án. Vì vậy nhóm chia theo nhóm gồm tác vụ, chiều chất lượng và mức bẩn, tổng cộng 42 nhóm, dùng 5-fold lặp với ba seed. Câu hỏi đánh giá trở thành: gặp một mức bẩn chưa từng thấy, mô hình đoán Δ đúng đến đâu. Chỉ số gồm MAE, Spearman và R². Để biết meta-model hơn baseline có thật hay không, nhóm dùng bootstrap theo nhóm, 1.000 lần, lấy khoảng tin cậy 95%. Ngoài ra có bài toán Good/Bad: lượt nào mất hơn 10% hiệu năng được coi là Bad, để so được với rule-based.”

## Slide 9 — RQ1: heatmap (1,5 phút, Mạnh)

“Câu hỏi thứ nhất: ảnh hưởng của chất lượng dữ liệu có phụ thuộc tác vụ không. Câu trả lời là có, và rất rõ. Hình này cho tỷ lệ hiệu năng bị mất ở mức bẩn 0,8. Với Completeness, regression mất 59 đến 64% hiệu năng, classification chỉ mất 21 đến 23%, còn clustering mất 73 đến 75%. Với Feature Accuracy, clustering mất tới 99%, AMI về gần 0. Chiều có ảnh hưởng lớn nhất cũng khác nhau: Completeness với regression, Feature Accuracy với clustering. Ngược lại, Uniqueness gần như không ảnh hưởng ở cả ba tác vụ, độ lớn của Δ không vượt 0,022. Trong cùng một tác vụ, thuật toán cũng quan trọng: Random Forest nhạy với Feature Accuracy hơn Logistic Regression.”

## Slide 10 — RQ1: đường cong suy giảm (1,5 phút, Mạnh)

“Hình này cho thấy Δ theo mức bẩn. Điểm đáng chú ý là Target Accuracy của classification không đơn điệu, nó có dạng chữ U ngược. Δ F1 của Random Forest đạt đỉnh 0,226 ở mức bẩn 0,5, rồi giảm về 0,098 ở mức 0,8, bằng mức ở 0,2. Nhóm giả thuyết rằng với hai lớp, đổi 80% nhãn ở cả train và test gần như đảo ngược nhãn một cách nhất quán, nên mô hình học lại được quy luật ngược; nhiễu lớn nhất là ở 0,5. Em xin nói rõ đây là giả thuyết nhất quán với số liệu, nhóm chưa kiểm chứng riêng. Hệ quả là một điểm chất lượng tuyến tính không thể biểu diễn quan hệ này.”

## Slide 11 — RQ2: meta-model so với baseline (1,5 phút, Mạnh)

“Câu hỏi thứ hai: meta-model có hơn các điểm chất lượng cố định không. Trên 312 lượt, meta-model có MAE 0,055, thấp hơn weighted score 0,077 và DQSOps-PCA 0,084, tức thấp hơn khoảng 28% và 35%. Về khả năng xếp hạng mức suy giảm, Spearman của meta-model là 0,66, weighted score 0,32, còn DQSOps-PCA chỉ 0,06, gần như không xếp hạng được. Meta-model có MAE thấp nhất và Spearman cao nhất ở cả ba tác vụ. Nhóm dùng MAE và Spearman làm thước đo chính vì R² trong từng tác vụ không ổn định: mỗi fold giấu trọn một số nhóm nên vài lỗi lớn kéo R² xuống âm.”

## Slide 12 — RQ2: bootstrap (1,5 phút, Mạnh)

“Hình này đặt Δ dự đoán cạnh Δ thật. Điểm của meta-model bám theo đường chéo, còn weighted score và DQSOps-PCA gần như đoán cùng một mức cho mọi lượt nên sai nặng ở những lượt regression có Δ lớn, tới 0,55. Về ý nghĩa thống kê, kết quả có hai mặt. Meta-model hơn DQSOps-PCA có ý nghĩa: hiệu MAE là âm 0,029, khoảng tin cậy từ âm 0,048 đến âm 0,011, không chứa 0. Nhưng so với weighted score, hiệu là âm 0,022, khoảng tin cậy từ âm 0,047 đến dương 0,005 có chứa 0, nên chưa đủ bằng chứng. Với chỉ 42 nhóm, sức mạnh thống kê có hạn, vì vậy nhóm không khẳng định meta-model vượt trội mọi baseline.”

## Slide 13 — RQ2: bài toán Good/Bad (1,5 phút, Mạnh)

“Khi chỉ cần trả lời dữ liệu đủ tốt hay không, bức tranh khác đi. Rule-based báo Bad cho 92,3% số lượt, vì Class Balance của dữ liệu sạch đã dưới ngưỡng 0,9 ở cả HIGGS và Covertype. Một luật ngưỡng cố định không thích nghi với mức nền tự nhiên của từng bộ dữ liệu. Điều bất ngờ là weighted score thắng ở bài toán này, F1 của lớp Bad là 0,790 so với 0,746 của meta-model. Cách hiểu của nhóm: điểm trung bình đủ để nhận ra có chiều nào đó bị bẩn, nhưng không đủ để biết hiệu năng giảm bao nhiêu. Khi nâng ngưỡng lên 20%, meta-model bền hơn, 0,687 so với 0,560. Tóm lại, trả lời câu hỏi hai: có, ở mức vừa phải.”

## Slide 14 — RQ3: SHAP so với PCA (1,5 phút, Mạnh)

“Câu hỏi thứ ba: trọng số có khác nhau theo tác vụ không. Bên phải là tỷ trọng SHAP của meta-model: chiều đứng đầu khác nhau, Feature Accuracy với classification và clustering, Completeness với regression, và Uniqueness thấp nhất ở cả ba tác vụ. Bên trái là trọng số của PCA, một bộ chung cho mọi tác vụ. PC1 chỉ giải thích 26,8% phương sai. Feature Accuracy, chiều làm giảm hiệu năng mạnh, lại có trọng số gần 0, và Target Accuracy mang trọng số âm, nghĩa là nhãn càng sai thì điểm chất lượng càng cao. Lý do là mỗi lượt chỉ làm bẩn một chiều nên các chiều gần như không tương quan, và PC1 chủ yếu nắm dòng này thuộc tác vụ nào chứ không nắm dữ liệu bẩn đến đâu. Một lưu ý khi đọc: Uniqueness có tỷ trọng 0,107 ở classification nhưng điều đó không có nghĩa bản ghi trùng làm giảm hiệu năng; mô hình dùng nó như tín hiệu loại trừ.”

## Slide 15 — RQ4: khả năng mở rộng (1,5 phút, Mạnh)

“Câu hỏi cuối: bước profiling có mở rộng được không. Nhóm chạy profiling trên HIGGS đầy đủ từ 0,1 đến 11 triệu dòng, và thêm mốc 22 triệu để tìm điểm pandas không còn chạy được. Hai engine cho cùng điểm profile ở mọi mốc, sai số cỡ 10 mũ trừ 16. Bộ nhớ của pandas tăng khoảng 0,65 GB mỗi triệu dòng: 11 triệu dòng cần 7,3 GB, 22 triệu dòng cần khoảng 14,4 GB, vượt 13,6 GB RAM của máy Colab nên pandas hết bộ nhớ. PySpark chạy trọn 22 triệu dòng trong giới hạn 4 GB, mất khoảng 527 giây. Đổi lại PySpark chậm hơn pandas từ 4 đến 38 lần ở các mốc pandas còn chạy được. Kết luận thực tế: dữ liệu vừa RAM thì dùng pandas, vượt RAM thì dùng PySpark. Giới hạn cần nói thẳng: Spark chạy chế độ local trên hai nhân, và mốc 22 triệu là HIGGS ghép hai lần.”

## Slide 16 — Ba khác biệt so với bài gốc (1,5 phút, Mạnh)

“Trong quá trình làm, nhóm có ba điểm khác bài gốc tác động lớn nhất đến kết quả. Một, mẫu số của công thức accuracy cho cột số: bài gốc chia cho trung bình của cột, nhưng HIGGS có cột đối xứng quanh 0 cho điểm âm 499 và cột âm cho điểm 1,25; nhóm chia cho trung bình trị tuyệt đối, và vẫn trùng bài gốc trên cột dương. Hai, chia train test theo nhóm để chống rò rỉ như đã nói. Ba, Class Balance trên dữ liệu nhiều lớp đi ngược chiều: ở Covertype điểm tăng từ 0,254 lên 0,451 khi làm bẩn hơn, trong khi AMI vẫn giảm. Nhóm giữ nguyên hiện tượng này và dùng nó làm phép thử, vì weighted score và DQSOps-PCA đoán sai hướng còn meta-model biết tác vụ nên học được chiều ngược. Bỏ 24 lượt này thì kết luận vẫn đứng vững.”

## Slide 17 — Hạn chế và hướng phát triển (1 phút, Mạnh)

“Về hạn chế: mỗi tác vụ chỉ có một bộ dữ liệu nên thông tin tác vụ lẫn với thông tin bộ dữ liệu; lỗi là lỗi tổng hợp, làm bẩn từng chiều riêng lẻ; meta-model học trên 312 dòng nên chưa đủ bằng chứng hơn weighted score; và Spark chạy local. Hướng phát triển: thêm nhiều bộ dữ liệu cho mỗi tác vụ và đánh giá bằng cách giấu cả một bộ, làm bẩn nhiều chiều cùng lúc với lỗi có cấu trúc, mở rộng ma trận thí nghiệm, và chạy profiling trên cụm Spark thật.”

## Slide 18 — Kết luận (1 phút, Nam)

“Em xin tóm tắt theo bốn câu hỏi. Một, ảnh hưởng của chất lượng dữ liệu thay đổi theo tác vụ và thuật toán, và Uniqueness gần như không ảnh hưởng. Hai, meta-model có MAE 0,055 và Spearman 0,66, hơn DQSOps-PCA có ý nghĩa thống kê. Ba, SHAP cho trọng số khác nhau theo tác vụ, còn trọng số PCA cố định không phản ánh mức ảnh hưởng. Bốn, PySpark cho cùng kết quả với pandas và chạy được 22 triệu dòng khi pandas hết bộ nhớ. Thông điệp chính của đề tài: một điểm chất lượng dữ liệu chỉ có ý nghĩa khi gắn với mục đích sử dụng. Em xin cảm ơn thầy cô và các bạn, mời thầy cô đặt câu hỏi.”

---

## Chuẩn bị cho phần hỏi đáp

**1. Vì sao không dùng đúng công thức accuracy của bài gốc?** Công thức gốc chia cho trung bình cột, ngầm giả định cột dương. HIGGS có cột đối xứng quanh 0 (điểm âm 499) và Beijing có cột âm (điểm 1,25). Nhóm chia cho trung bình trị tuyệt đối; với cột dương hai công thức trùng nhau, nên đây là mở rộng chứ không phải thay công thức.

**2. 312 dòng có đủ cho meta-model không?** Chưa chắc, nhóm nói rõ điều này: khoảng tin cậy rộng, chưa đủ bằng chứng hơn weighted score. Vì vậy nhóm giữ siêu tham số cố định, dùng cây nông, đánh giá theo nhóm và báo kèm độ lệch chuẩn qua ba seed.

**3. Vì sao weighted score thắng ở Good/Bad?** Điểm trung bình đủ để phát hiện có chiều nào đó bị bẩn, nhưng không ước lượng được bẩn đến mức giảm bao nhiêu hiệu năng; phần định lượng cần biết tác vụ. Gợi ý dùng thực tế: dùng điểm trung bình để phát hiện, dùng meta-model để định lượng.

**4. Spark local trên 2 nhân có phải “quy mô lớn” không?** Không phải cụm thật; nhóm chỉ minh họa xu hướng và chỉ ra điểm pandas hết RAM (22 triệu dòng trên máy 13,6 GB). Nhóm đã nêu giới hạn này, và nêu chạy trên cụm thật là hướng phát triển.

**5. Nếu bỏ 24 lượt Class Balance × clustering thì sao?** Kết luận không đổi: MAE tổng thể 0,057 so với 0,079 và 0,088, và khoảng cách ở clustering còn lớn hơn. Nhóm giữ đủ 312 lượt trong bảng chính vì bỏ đúng những lượt baseline đoán sai sẽ giống chọn dữ liệu có lợi.

**6. Vì sao R² trong từng tác vụ lại âm?** Mỗi fold giấu trọn một số nhóm (ví dụ mọi lượt ở mức bẩn 0,8 của một chiều), trong khi phương sai của Δ trong một tác vụ nhỏ, nên vài lỗi lớn kéo R² xuống âm. Vì vậy MAE và Spearman là thước đo chính.

**7. Giả thuyết chữ U ngược của Target Accuracy đã kiểm chứng chưa?** Chưa. Đây là giải thích nhất quán với dữ liệu; nhóm nói rõ là chưa kiểm chứng riêng.

**8. Vì sao chỉ có 5 chiều, bỏ Consistent Representation?** Để thu gọn phạm vi đồ án; chiều này cũng không áp dụng được cho HIGGS vì bộ này toàn cột số.
