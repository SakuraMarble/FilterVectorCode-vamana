### 把json的属性文件变成txt属性文件

import json

# 读取 JSON 文件
with open('/data/fxy/FilterVector/FilterVectorData/amazing_file/amazing_logos_text_keywords_ids.json', 'r') as file:
    data = json.load(file)

# 打开一个 TXT 文件，准备写入
with open('/data/fxy/FilterVector/FilterVectorData/amazing_file/amazing_file_base_labels.txt', 'w') as output_file:
    for vector in data:
        # 将每个向量的元素连接为逗号分隔的字符串
        line = ",".join(map(str, vector))
        output_file.write(line + '\n')

