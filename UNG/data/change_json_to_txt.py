### 把json的属性文件变成txt属性文件

import json

# 读取 JSON 文件
with open('/home/fengxiaoyao/FilterVector/FilterVectorData/tiktok_reviews_attr.json', 'r') as file:
    data = json.load(file)

# 打开一个 TXT 文件，准备写入
with open('/home/fengxiaoyao/FilterVector/FilterVectorData/tiktok_reviews_attr.txt', 'w') as output_file:
    for vector in data:
        # 将每个向量的元素连接为逗号分隔的字符串
        line = ",".join(map(str, vector))
        output_file.write(line + '\n')

