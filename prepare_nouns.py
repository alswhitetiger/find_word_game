import re
from gensim.models import FastText

print("AI 모델을 로딩하여 깨끗한 명사 목록을 추출합니다.")
print("이 과정은 몇 분 정도 소요될 수 있습니다...")

try:
    model = FastText.load_fasttext_format('cc.ko.300.bin')
except FileNotFoundError:
    print("\n[오류] 'cc.ko.300.bin' 파일을 찾을 수 없습니다.")
    exit()

blacklist = ['것', '수', '그', '이', '저', '등', '때', '곳', '자신']
korean_pattern = re.compile("^[가-힣]+$")

filtered_nouns = []
for word in model.wv.index_to_key:
    if korean_pattern.match(word) and word not in blacklist and 1 < len(word) < 7:
        filtered_nouns.append(word)

with open('filtered_noun_list.txt', 'w', encoding='utf-8') as f:
    for word in filtered_nouns:
        f.write(word + '\n')

print(f"\n완료! 깨끗한 명사 {len(filtered_nouns):,}개를 'filtered_noun_list.txt' 파일에 저장했습니다.")