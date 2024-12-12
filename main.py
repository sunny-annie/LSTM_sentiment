import streamlit as st
import torch
from LSTM_model import predict_sentiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.markdown("""
## 🧠 LSTM против кинокритиков 🍿  
Этот проект — как кинокритик, но с искусственным интеллектом! 
LSTM-нейросеть обучена читать отзывы пользователей с IMDB и решать, где фильм — шедевр, а где — полный провал.  

📚 **Что внутри:**  
- Датасет отзывов с IMDB 🎬  
- Нейросеть, которая считает, что она разбирается в кино  
- Немного магии NLP и PyTorch ✨  

⚠️ **Внимание:** нейросеть отлично понимает только отзывы на английском языке. Так что никаких "шедевр, 10 из 10" или 
"невероятно скучно, 2/10" — она этого не оценит. 😅 Только *"Amazing movie, loved every second!"* или *"Terrible script 
and poor acting. Not worth watching."*

Попробуйте сами и узнайте, что ИИ думает об отзывах на ваши любимые фильмы! 🚀  
""")

txt_label = 'Введите текст отзыва на английском языке:'
default_text = "No one is more disappointed than me. I don't know how they managed to make Dune boring. " \
               "I can only assume it's intentional because a miss this wide is impossible to attribute to incompetence."

txt = st.text_area(label=txt_label, value=default_text, height=200)

with st.form('button'):
    button_click = st.form_submit_button("Узнать результат", use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    pass

with col2:
    st.subheader("Прогноз:")
    if button_click:
        st.write(f'`{predict_sentiment(txt)}`')

with col3:
    pass