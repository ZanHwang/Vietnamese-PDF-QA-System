M1_PROMPT = """Dựa vào các đoạn văn bản dưới đây, hãy trả lời câu hỏi một cách ngắn gọn và chính xác bằng tiếng Việt.
Chỉ dựa vào thông tin trong ngữ cảnh được cung cấp. Nếu không tìm thấy câu trả lời, hãy nói "Không tìm thấy thông tin."

Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời:"""

QUERY_EXPANSION_PROMPT = """Hãy sinh ra {n} cách diễn đạt khác nhau cho câu hỏi sau bằng tiếng Việt.
Mỗi cách diễn đạt trên một dòng, không đánh số, không giải thích.

Câu hỏi gốc: {question}

Các cách diễn đạt khác:"""

# M4 — CRAG
CRAG_RELEVANCE_PROMPT = """Đánh giá độ liên quan của từng đoạn văn bản với câu hỏi.
Với mỗi đoạn trả lời đúng một nhãn: CORRECT / AMBIGUOUS / INCORRECT
- CORRECT: đoạn rõ ràng chứa thông tin để trả lời câu hỏi
- AMBIGUOUS: đoạn có thể liên quan nhưng không chắc chắn
- INCORRECT: đoạn không liên quan đến câu hỏi

Câu hỏi: {question}

{chunks}

Trả lời theo định dạng (mỗi dòng một nhãn, không giải thích):
Đoạn 1: CORRECT/AMBIGUOUS/INCORRECT
Đoạn 2: CORRECT/AMBIGUOUS/INCORRECT"""

CRAG_FILTER_PROMPT = """Từ đoạn văn bản dưới đây, hãy trích xuất chỉ các câu trực tiếp liên quan đến câu hỏi.
Giữ nguyên câu gốc, không thêm thông tin. Nếu không có câu nào liên quan, trả lời "NONE".

Câu hỏi: {question}
Đoạn văn: {chunk}

Các câu liên quan:"""

# M5 — Self-RAG Lite
SELFRAG_RETRIEVE_PROMPT = """Câu hỏi sau có cần tra cứu tài liệu không?
Trả lời YES nếu cần thông tin cụ thể (sự kiện, số liệu, tên, ngày tháng, địa danh...).
Trả lời NO nếu có thể trả lời từ kiến thức chung.
Chỉ trả lời YES hoặc NO, không giải thích.

Câu hỏi: {question}"""

SELFRAG_REFLECT_PROMPT = """Câu trả lời sau có được hỗ trợ hoàn toàn bởi ngữ cảnh không?
Trả lời SUPPORTED nếu mọi thông tin trong câu trả lời đều có trong ngữ cảnh.
Trả lời NOT_SUPPORTED nếu câu trả lời chứa thông tin không có trong ngữ cảnh.
Chỉ trả lời SUPPORTED hoặc NOT_SUPPORTED, không giải thích.

Ngữ cảnh: {context}
Câu trả lời: {answer}"""

SELFRAG_STRICT_PROMPT = """Dựa CHÍNH XÁC vào các đoạn văn bản dưới đây, hãy trả lời câu hỏi ngắn gọn.
KHÔNG được thêm bất kỳ thông tin nào ngoài ngữ cảnh. Nếu không tìm thấy, hãy nói "Không tìm thấy thông tin."

Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời (chỉ dựa vào ngữ cảnh):"""
