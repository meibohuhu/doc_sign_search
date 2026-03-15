import csv
import re
import sys

# 增加 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)


def clean_sentence(text):
    """
    清理句子：
    1. 去掉所有 [] 格式的内容（包括方括号）
    2. 去掉所有 () 格式的内容（包括圆括号，如 (expressive gesture)）
    3. 去掉 www.xxx 格式的内容（包括www.后面的域名，不区分大小写）
    4. 去掉 http:// 或 https:// 开头的链接
    5. 去掉 &gt; 或 &gt HTML实体
    6. ghlUhZ5L7AY	这种多人对话的，later clean up
    7. Email: dhs.dhhsd@state.mn.us 这种email地址的

    """
    if not text:
        return text
    
    # 去掉所有方括号及其内容 [xxx]
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # 去掉所有圆括号及其内容 (xxx)，如 (expressive gesture)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 去掉独立的 www.xxx 格式（包括可能的空格和标点，不区分大小写）
    # 匹配 www. 后面跟域名，可能前面有空格或标点，可能被方括号包围
    text = re.sub(r'\s*\[?\s*www\.[^\s\[\](){}"\']+\s*\]?\s*', ' ', text, flags=re.IGNORECASE)
    
    # 去掉 http:// 或 https:// 开头的链接（包括后面的URL，也包括单独的 http:// 或 https://）
    text = re.sub(r'https?://[^\s]*', '', text, flags=re.IGNORECASE)
    
    # 去掉 HTML 实体 &gt; 和 &gt
    text = re.sub(r'&gt;?', '', text, flags=re.IGNORECASE)
    
    # 去掉 email 地址（匹配 Email: xxx@xxx.com 或直接的 xxx@xxx.com 格式）
    # 先去掉 "Email:" 或 "email:" 后面跟的email地址
    text = re.sub(r'\s*[Ee]mail\s*:\s*[^\s@]+@[^\s@]+\.[^\s@]+', '', text, flags=re.IGNORECASE)
    # 再去掉单独的 email 地址格式
    text = re.sub(r'\s*[^\s@]+@[^\s@]+\.[^\s@]+', '', text)
    
    # 去掉多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_csv_file(input_file, output_file):
    """清理CSV文件中的SENTENCE字段"""
    print(f"正在处理: {input_file}")

    cleaned_count = 0
    total_count = 0
    output_rows = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_count += 1
            original_sentence = row.get('SENTENCE', '')
            cleaned_sentence = clean_sentence(original_sentence)
            
            # 如果句子被清理了，记录下来
            if cleaned_sentence != original_sentence:
                cleaned_count += 1
            row['SENTENCE'] = cleaned_sentence
            
            # 如果清理后的句子不为空，保留这一行
            if cleaned_sentence:
                output_rows.append(row)

    print(f"  总行数: {total_count}")
    print(f"  清理的行数: {cleaned_count}")
    print(f"  输出行数: {len(output_rows)}")

    # 写入输出文件
    print(f"正在写入: {output_file}")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"完成！已写入 {len(output_rows)} 行到 {output_file}\n")
    return cleaned_count, total_count, len(output_rows)


def main():
    files_to_clean = [
        ("/home/mh2803/projects/sign_language_llm/youtube_asl_clips_stage2.csv",
         "/home/mh2803/projects/sign_language_llm/youtube_asl_clips_stage2.csv"),
        ("/home/mh2803/projects/sign_language_llm/youtube_asl_clips.csv",
         "/home/mh2803/projects/sign_language_llm/youtube_asl_clips.csv"),
    ]
    
    total_cleaned = 0
    total_rows = 0
    total_output = 0
    
    for input_file, output_file in files_to_clean:
        cleaned, total, output = clean_csv_file(input_file, output_file)
        total_cleaned += cleaned
        total_rows += total
        total_output += output
    
    print("=" * 50)
    print(f"总计:")
    print(f"  清理的行数: {total_cleaned}")
    print(f"  总行数: {total_rows}")
    print(f"  输出行数: {total_output}")


if __name__ == "__main__":
    main()
