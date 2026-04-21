# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 51
- **Tỉ lệ Pass/Fail:** 51 Pass / 0 Fail
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.1557
    - Relevancy: 0.1215
    - Hit Rate: 0.9020
    - MRR: 0.7549
- **Điểm LLM-Judge trung bình:** 3.73 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination / OOC | ~10 | Câu hỏi cố tình ngoài ngữ cảnh (Out Of Context) hoặc yêu cầu làm thơ, khiến lòng tin (Faithfulness) = 0 và mức độ liên quan thấp vì Agent bị lạc đề hoặc không có thông tin trong Context. |
| Missing Information | ~5 | Retriever tìm thấy Document (Hit Rate cao) nhưng do Chunking size chưa tốt hoặc Ingestion không bắt được nội dung sâu, khiến Agent trả lời không đủ ý. |
| Prompt Injection | ~3 | Câu hỏi dạng lừa Agent bỏ qua Context khiến điểm Relevancy bằng 0 vì Agent từ chối trả lời hoặc bị cướp quyền điều khiển (Goal Hijacking). |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: What is 'Prompt Injection'?
1. **Symptom:** Agent có điểm RAGAS Relevancy = 0, mặc dù lấy được đúng Doc.
2. **Why 1:** LLM trả lời quá ngắn hoặc không miêu tả đầy đủ ý nghĩa theo yêu cầu của Judge.
3. **Why 2:** Vector DB tìm thấy tài liệu liên quan nhất nhưng đoạn Chunk chứa định nghĩa Prompt Injection thiếu ngữ cảnh bổ sung.
4. **Why 3:** Chunking size có thể đang dùng Fixed-size cố định, vô tình cắt đứt mạch câu của định nghĩa.
5. **Why 4:** System Prompt của Agent chưa yêu cầu giải thích cặn kẽ và chi tiết theo phong cách của Judge.
6. **Root Cause:** Chiến lược Chunking chưa tối ưu (cắt ngang ý) và System Prompt thiếu hướng dẫn về độ sâu của câu trả lời.

### Case #2: What steps are involved in the final submission checklist...
1. **Symptom:** Điểm Faithfulness và Relevancy đều bằng 0, thư viện đánh giá thất bại trong việc tìm thấy thông tin hữu ích trong câu trả lời.
2. **Why 1:** Agent trả lời: "Dựa trên các đoạn tài liệu được cung cấp, không có thông tin chi tiết nào về các bước cụ thể".
3. **Why 2:** Dù Hit Rate = 1.0 (nhặt được tài liệu), nhưng tài liệu chứa Checklist không được cung cấp rõ ràng.
4. **Why 3:** Quá trình Ingestion chưa trích xuất hoặc nhận diện được cấu trúc danh sách (List/Checklist) từ tài liệu gốc.
5. **Why 4:** Trình parse PDF hoặc text chưa bảo toàn được format markdown/list của file quy chế.
6. **Root Cause:** Ingestion Pipeline làm mất cấu trúc danh sách liệt kê, khiến LLM không nhìn thấy các bước cụ thể.

### Case #3: What is the maximum score for the Failure Analysis criterion?
1. **Symptom:** Điểm Faithfulness và Relevancy đều bằng 0 mặc dù Agent trả lời đúng điểm tối đa là 5.
2. **Why 1:** Công cụ RAGAS chấm điểm thấp do không khớp chính xác với tiêu chí Ground Truth hoặc cách hành văn.
3. **Why 2:** Agent trả lời bằng tiếng Anh ("The maximum score... is 5"), trong khi tài liệu và Ground Truth có thể là tiếng Việt.
4. **Why 3:** System Prompt không ép buộc hoặc quy định ngôn ngữ phản hồi (Language Consistency).
5. **Why 4:** Không có cơ chế nhận diện ngôn ngữ của người dùng để trả lời tương ứng (hoặc hệ thống mặc định dịch sang tiếng Anh).
6. **Root Cause:** Không nhất quán về ngôn ngữ (Language mismatch) giữa Agent, Câu hỏi và Metric chấm điểm.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking để bảo toàn ngữ nghĩa các định nghĩa và List/Checklist.
- [ ] Cập nhật System Prompt để nhấn mạnh: "Luôn phản hồi bằng ngôn ngữ của câu hỏi" (Language Consistency).
- [ ] Bổ sung cơ chế xử lý ngoại lệ (Out Of Context) một cách chuyên nghiệp hơn: Nếu không có trong Context, trả lời khuôn mẫu thay vì cố suy luận.
- [ ] Nâng cấp Ingestion Pipeline để giữ được định dạng bảng biểu và danh sách liệt kê.
- [ ] Thêm bước Reranking vào Pipeline để đẩy các đoạn Chunk chứa định nghĩa/checklist lên vị trí đầu tiên.
