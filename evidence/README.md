# Bằng chứng Lab Day 22 - LangSmith + Prompt Versioning

Sinh viên: Nguyễn Anh Đức - MSSV 2A202601930

## Cấu hình hệ thống

Mô hình sinh câu trả lời là openai/gpt-oss-20b chạy qua Groq. Mô hình chấm điểm RAGAS là gemini-3.5-flash-lite của Google AI Studio. Embedding dùng BAAI/bge-small-en-v1.5 chạy cục bộ bằng fastembed để không tiêu tốn hạn mức API. Retriever lấy 3 đoạn văn bản cho mỗi câu hỏi, knowledge base được chia thành 107 chunk.

Cả hai phiên bản prompt được chấm bởi cùng một mô hình đánh giá với cùng bộ 50 cặp câu hỏi và cùng cấu hình retriever, nên bảng so sánh dưới đây là so sánh có kiểm soát.

## Kết quả bốn chỉ số

| Chỉ số            | Prompt V1 | Prompt V2 |
| ----------------- | --------- | --------- |
| faithfulness      | 0.9550    | 0.8037    |
| answer_relevancy  | 0.9449    | 0.9298    |
| context_recall    | 1.0000    | 1.0000    |
| context_precision | 0.9633    | 0.9633    |

Cả hai phiên bản đều vượt ngưỡng faithfulness 0.8 mà đề bài yêu cầu.

## Phân tích vì sao V1 cao hơn V2

Khác biệt nằm ở thiết kế của hai system prompt. V1 yêu cầu câu trả lời ngắn gọn trong 2 đến 4 câu và cấm bổ sung thông tin ngoài context, kèm một câu trả lời mặc định khi context không chứa đáp án. V2 đóng vai chuyên gia phân tích kỹ thuật, yêu cầu viết 3 đến 5 câu có tổ chức và mạch lạc.

Faithfulness đo tỷ lệ các mệnh đề trong câu trả lời được context hỗ trợ. Khi prompt yêu cầu viết dài hơn và trình bày có cấu trúc, mô hình có xu hướng bổ sung câu chuyển ý, câu khái quát hóa và kiến thức nền sẵn có để làm câu trả lời trôi chảy. Những mệnh đề thêm vào này không nằm trong ba đoạn context được truy xuất, nên bị tính là không được hỗ trợ. Đây chính là lý do V2 tụt xuống 0.8037 trong khi V1 giữ được 0.9550. Nói cách khác, ràng buộc độ dài chặt của V1 hoạt động như một hàng rào chống bịa đặt.

Answer relevancy chênh lệch không đáng kể, 0.9449 so với 0.9298, cho thấy cả hai prompt đều bám sát câu hỏi. Phần chênh nhỏ đến từ việc câu trả lời dài của V2 chứa thêm nội dung không trực tiếp trả lời câu hỏi.

Context recall và context precision bằng nhau tuyệt đối giữa hai phiên bản. Điều này đúng về mặt nguyên lý, vì hai chỉ số này chỉ đánh giá chất lượng của bước truy xuất, mà bước truy xuất dùng chung một vectorstore, chung một câu hỏi và chung tham số k bằng 3. System prompt chỉ tác động tới bước sinh câu trả lời, không tác động tới retriever. Sự trùng khớp này là một kiểm chứng cho thấy pipeline đánh giá đang hoạt động đúng chứ không phải trùng hợp ngẫu nhiên.

Kết luận rút ra là với một hệ thống RAG cần độ tin cậy cao, prompt ràng buộc ngắn gọn và cấm suy diễn ngoài context cho kết quả tốt hơn prompt khuyến khích diễn giải mở rộng, ngay cả khi phiên bản thứ hai đọc mượt hơn với người dùng.

## Hạn chế cần ghi nhận

Chỉ số faithfulness của V2 được tính trên 49 trên tổng số 50 mẫu. Một mẫu bị bỏ do mô hình đánh giá trả về chuỗi JSON chứa dấu nháy đơn chưa escape khiến bước parse thất bại. Đây là lỗi của tầng đánh giá, không phải lỗi của pipeline RAG, và cả 50 cặp câu hỏi đều đã được chạy qua cả hai phiên bản prompt. Ba chỉ số còn lại của V2 và toàn bộ bốn chỉ số của V1 đều tính đủ 50 mẫu.

## Danh sách file bằng chứng

| File                    | Nội dung                                      |
| ----------------------- | --------------------------------------------- |
| 01_langsmith_traces.png | Giao diện LangSmith với 51 run được chọn      |
| 01_rag_pipeline_log.txt | Log console của bước 1                        |
| 02_prompt_hub.png       | Prompt Hub hiển thị hai phiên bản prompt      |
| 02_ab_routing_log.txt   | Log A/B routing, 50 truy vấn có nhãn v1 và v2 |
| 03_ragas_scores.png     | Bảng so sánh bốn chỉ số của V1 và V2          |
| 03_ragas_report.json    | Báo cáo JSON đầy đủ                           |
| 03_ragas_run_log.txt    | Log console của bước 3                        |
| 04_pii_demo_log.txt     | Sáu test case của PII detector                |
| 04_json_demo_log.txt    | Sáu test case của JSON formatter              |
