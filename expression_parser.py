# BƯỚC 1 — LÀM SẠCH
def clean_characters(raw_chars):
    # 1. Xóa các mục trống.
    chars = [c.strip() for c in raw_chars if c.strip() != '']

    # 2. Chỉ gộp các toán tử trùng lặp chính xác liên tiếp
    operators = {'+', '-', '*', '÷'}
    cleaned = []
    for c in chars:
        if cleaned and c in operators and cleaned[-1] == c:
            continue  
        cleaned.append(c)

    return cleaned

# BƯỚC 2 — XÁC THỰC
def validate_expression(char_list):
    if not char_list:
        return False, "Expression is empty."

    expr = ''.join(char_list)

    # Kiểm tra bắt đầu / kết thúc
    if char_list[0] in ('+', '*', '÷', ')'):
        return False, f"Expression cannot start with '{char_list[0]}'."
    if char_list[-1] in ('+', '-', '*', '÷', '('):
        return False, f"Expression cannot end with '{char_list[-1]}'."

    # toán tử liên tiếp
    binary_ops = {'+', '-', '*', '÷'}
    for i in range(1, len(char_list)):
        if char_list[i] in binary_ops and char_list[i-1] in binary_ops:
            #  Cho phép: '(' theo sau bởi '-'
            if not (char_list[i] == '-' and char_list[i-1] == '('):
                return False, f"Two consecutive operators '{char_list[i-1]}' and '{char_list[i]}' at position {i}."

    #  Cân bằng dấu ngoặc 
    depth = 0
    for i, c in enumerate(char_list):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth < 0:
            return False, f"Unmatched ')' at position {i}."
    if depth != 0:
        return False, "Unmatched '(' — parentheses are not balanced."

    # Empty parentheses
    if '()' in expr:
        return False, "Empty parentheses '()' found."

    return True, "OK"

def build_and_evaluate(raw_chars):
    cleaned = clean_characters(raw_chars)
    expression_str = ''.join(cleaned)

    is_valid, msg = validate_expression(cleaned)
    if not is_valid:
        return expression_str, None, f"Validation failed: {msg}"
    return expression_str, None, None