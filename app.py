import streamlit as st
import random
from gensim.models import FastText

@st.cache_resource
def load_data():
    try:
        model = FastText.load_fasttext_format('cc.ko.300.bin')
        with open('filtered_noun_list.txt', 'r', encoding='utf-8') as f:
            noun_vocab = [line.strip() for line in f]
        return model, noun_vocab
    except FileNotFoundError:
        return None, None

def generate_new_quiz():
    a, b, c, predicted_d = None, None, None, None
    for _ in range(100):
        try:
            a = random.choice(noun_vocab)
            similar_to_a = model.wv.most_similar(a, topn=50)
            
            candidates_c = [word for word, sim in similar_to_a if sim > 0.5 and word in noun_vocab and word != a]
            if not candidates_c: continue
            c = random.choice(candidates_c)

            candidates_b = [word for word, sim in similar_to_a if sim > 0.6 and word in noun_vocab and word != a and word != c]
            if not candidates_b: continue
            b = random.choice(candidates_b)

            top_predictions = model.wv.most_similar(positive=[c, b], negative=[a], topn=10)
            
            for word, _ in top_predictions:
                if word in noun_vocab:
                    predicted_d = word
                    break 
            
            if all([a, b, c, predicted_d]):
                break
        except (KeyError, IndexError):
            continue
            
    if not all([a, b, c, predicted_d]):
        st.session_state.quiz_generated_successfully = False
        return

    st.session_state.quiz = {'a': a, 'b': b, 'c': c, 'predicted_d': predicted_d}
    st.session_state.quiz_generated_successfully = True
    st.session_state.user_answer_input = ""

# ✨ 1. 입력창을 초기화할 콜백 함수를 만듭니다.
def clear_input():
    st.session_state.user_answer_input = ""

model, noun_vocab = load_data()

st.title("🧠 단어 유추 퀴즈")

if model is None or noun_vocab is None:
    st.error("퀴즈에 필요한 파일('cc.ko.300.bin' 또는 'filtered_noun_list.txt')이 없습니다.")
    st.info("'prepare_nouns.py'를 먼저 실행했는지 확인해주세요.")
    st.stop()

if 'quiz' not in st.session_state:
    st.session_state.quiz = {}
if 'quiz_generated_successfully' not in st.session_state:
    st.session_state.quiz_generated_successfully = True

if st.button("새로운 문제 만들기", type="primary", on_click=generate_new_quiz):
    pass # on_click 콜백을 사용하므로 버튼 아래 로직은 비워둡니다.

if not st.session_state.quiz_generated_successfully:
    st.warning("적절한 문제를 생성하지 못했습니다. 버튼을 다시 눌러주세요.")

if st.session_state.quiz and st.session_state.quiz_generated_successfully:
    q = st.session_state.quiz
    
    st.subheader(f"문제: `{q['a']}` : `{q['b']}` = `{q['c']}` : ?")

    user_answer = st.text_input("정답을 입력하세요:", key="user_answer_input")

    if st.button("정답 확인"):
        if user_answer:
            try:
                similarity = model.wv.similarity(user_answer, q['predicted_d'])
                st.metric(label=f"예측({q['predicted_d']})과의 유사도", value=f"{similarity:.2f}")
                
                if similarity > 0.6:
                    st.success("🎉 정답입니다!")
                elif similarity > 0.3:
                    st.info("🤔 아쉽네요!")
                else:
                    st.error("😥 틀렸습니다.")
            except KeyError:
                st.error(f"'{user_answer}'는 사전에 없는 단어입니다.")
        else:
            st.warning("정답을 입력해주세요.")

    col1, col2 = st.columns(2)

    with col1:
        # ✨ 2. '1순위 정답' 버튼에 on_click 콜백을 연결합니다.
        if st.button("1순위 정답", on_click=clear_input):
            st.info(f"모델이 예측한 1순위 정답은 **'{q['predicted_d']}'** 입니다.")
    
    with col2:
        # ✨ 3. '다음 문제' 버튼도 on_click 콜백을 사용하도록 수정합니다.
        if st.button("다음 문제", on_click=generate_new_quiz):
            pass