def feedbackSentimentAnalysis(result):
    if result == 0:
        return "Tiêu Cực"
    elif result == 1:
        return "Trung Lập"
    elif result == 2:
        return "Tích Cực"

def feedbackTopic(result):
    if result == 0:
        return "Giảng Viên"
    elif result == 1:
        return "Chương trình học"
    elif result == 2:
        return "Trang thiết bị"
    elif result == 3:
        return "Khác"