# CLAUDE.md

Hướng dẫn và quy tắc cho Claude khi làm việc trong project `data-quality-project` (đồ án môn Xử Lý Data). Đọc file này đầu tiên khi bắt đầu một phiên làm việc mới trên project này.

## Bắt đầu phiên làm việc

Trước khi làm bất kỳ việc gì, đọc:
1. `docs/gioi-thieu-tong-quan-de-tai.md` — tổng quan đề tài
2. `docs/checklist-trien-khai.md` — đang ở giai đoạn nào, việc gì đã xong/chưa xong

Không giả định tiến độ — luôn kiểm tra checklist trước khi nói project "đã" hay "chưa" làm gì.

## Ngôn ngữ

Giao tiếp bằng **tiếng Việt** trong mọi trao đổi, trừ khi được yêu cầu khác.

## Quy tắc an toàn — tuyệt đối tuân thủ

- **Trước khi tải file, cài thư viện/phần mềm, hoặc chỉnh sửa/xóa bất kỳ thứ gì thật trên máy local, PHẢI hỏi xác nhận trước.** Không tự ý thực hiện.
- **Trước khi xóa bất kỳ file nào, phải hỏi và được xác nhận rõ ràng trước khi xóa.**
- Không xóa file hệ thống/hệ điều hành, hay bất kỳ file nào có thể khiến máy mất dữ liệu hoặc bị treo (freeze). Chỉ được đụng tới file không liên quan đến hệ thống/OS, và chỉ sau khi đã xác nhận cụ thể.
- Không gỡ cài đặt hoặc xóa Visual Studio Code.
- Máy local là **laptop Windows cá nhân** (không phải máy công ty) — nhưng vẫn cần cẩn trọng như trên.

## Nguyên tắc hạ tầng

- **Google Colab** dùng cho các việc nặng CPU/dung lượng: tải dataset lớn (đặc biệt HIGGS full ~2.8GB), chạy Experiment Runner (312 lượt), Spark scalability study. Không tải/lưu các dataset lớn trực tiếp lên ổ cứng máy Windows — ổ đĩa đang hạn chế dung lượng.
- **VS Code (local)** dùng để viết code từng module trong `src/`, file config nhẹ (`configs/*.yaml`), chỉnh sửa tài liệu/báo cáo.
- Notebook khung Colab đã có sẵn tại `notebooks/data_quality_project_colab_setup.ipynb`.

## Quy tắc code & dữ liệu

- **Luôn đọc seed từ `configs/seeds.yaml`**, không hardcode số seed rải rác trong code — đảm bảo mọi kết quả tái lập được.
- **Không hardcode tham số thí nghiệm** (mức độ ô nhiễm, ma trận dataset × dimension × thuật toán) — luôn đọc từ `configs/`.
- `data/raw/` giữ nguyên bản gốc, **không bao giờ sửa trực tiếp**. Có lỗi thì xử lý lại từ `raw/`, không ghi đè.
- **Không tạo `data/polluted/`** — dữ liệu đã làm bẩn luôn tạo in-memory tại thời điểm chạy (nhờ seed cố định, tái lập được), không persist ra đĩa.
- Cột trong bảng kết quả (`results/experiment_results.parquet`) phải đúng theo schema đã định trong `docs/thiet-ke-thi-nghiem-chi-tiet.md` mục 5 — không tự ý đổi tên cột giữa các module.
- Hai bản cài đặt profiling (pandas và PySpark trong `src/profiling/`) phải cho **cùng công thức toán học**, chỉ khác engine thực thi.

## Quy ước đặt tên

- File Python: `snake_case.py`
- File kết quả: tên có tiền tố mô tả rõ nội dung (vd `experiment_results.parquet`, không đặt `results.parquet` chung chung)
- Không tự ý đổi cấu trúc thư mục đã chốt — xem `docs/cau-truc-thu-muc-project.md`

## Khi hoàn thành một việc

- Tick `[x]` vào đúng dòng tương ứng trong `docs/checklist-trien-khai.md`, giữ nguyên các mục chưa làm là `[ ]`.
- Không tự ý đánh dấu hoàn thành nếu chưa thực sự chạy/kiểm chứng được kết quả.
- **Mỗi khi hoàn thành một giai đoạn hoặc một mục checklist cụ thể, phải cập nhật lại TẤT CẢ các file `.md` liên quan**, đặc biệt là `docs/checklist-trien-khai.md` (tick `[x]` và cập nhật dòng "Trạng thái tổng quan"). Các file khác cần rà soát và cập nhật nếu bị ảnh hưởng: `docs/gioi-thieu-tong-quan-de-tai.md` (mục "Trạng thái hiện tại"), `README.md` gốc, và các `README.md` trong thư mục con liên quan (vd `data/raw/*/README.md`, `src/*/README.md`, `configs/README.md`, `results/README.md`), cùng các file thiết kế trong `docs/` nếu nội dung thực tế khác với thiết kế ban đầu.

- **Mỗi khi xong một giai đoạn hoặc một mục checklist, phải tường thuật lại cho người dùng ngay trong chat: đã làm gì và làm như thế nào** (các bước, quyết định thiết kế và lý do, kết quả kiểm chứng, sự cố nếu có, việc chưa làm). Đồng thời ghi lại đầy đủ vào `docs/cach-trien-khai-chi-tiet-tung-giai-doan.md` (thêm/cập nhật đúng mục của giai đoạn đó, ghi rõ phần nào chưa chạy/kiểm chứng).
- **Mỗi khi phát sinh điểm khác biệt so với hai bài báo gốc** (đổi công thức, đổi cách làm bẩn, đổi phạm vi, tự chốt một quy ước bài gốc không nêu), phải ghi vào `docs/khac-biet-so-voi-bai-goc.md` kèm lý do và trạng thái — file này là cơ sở làm slide giữa kỳ/cuối kỳ. Điểm lệch về công thức toán phải hỏi người dùng trước khi áp dụng, không tự quyết.
- Khi sắp sang bước cần model/mức khác (vd Sonnet Low → Medium → Opus/High), nhắc người dùng đổi trước khi làm.

## Tài liệu tham chiếu đầy đủ

| File | Nội dung |
|---|---|
| `docs/gioi-thieu-tong-quan-de-tai.md` | Tổng quan đề tài, vấn đề, đóng góp, phạm vi |
| `docs/tong-quan-bai-goc.md` | Tổng quan 2 bài báo gốc + khoảng trống nghiên cứu |
| `docs/pipeline-thi-nghiem.md` | Pipeline tổng thể, 10 module |
| `docs/thiet-ke-thi-nghiem-chi-tiet.md` | Ma trận thí nghiệm, công thức, schema kết quả |
| `docs/kien-truc-mo-hinh-va-ky-thuat.md` | Kiến trúc kỹ thuật từng module, thư viện, siêu tham số |
| `docs/cau-truc-thu-muc-project.md` | Cấu trúc thư mục đầy đủ + lý do thiết kế |
| `docs/checklist-trien-khai.md` | Checklist tiến độ theo từng giai đoạn |
| `docs/khac-biet-so-voi-bai-goc.md` | Mọi thay đổi/khác biệt so với 2 bài báo gốc, kèm lý do — cơ sở làm slide |
| `docs/cach-trien-khai-chi-tiet-tung-giai-doan.md` | Nhật ký: đã làm gì và làm như thế nào cho từng mục checklist |
