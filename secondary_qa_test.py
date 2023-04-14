import json
import pandas as pd
from deeppavlov import build_model, configs


# Constants
LINE = 100*'-'
DOUBLE_LINE = 100*'='
ARTICLES = [
            "Važenje krivičnog zakonodavstva na teritoriji Srbije \
            Član 6 \
            (1) Krivično zakonodavstvo Republike Srbije važi za svakog ko na njenoj teritoriji učini krivično delo. \
            (2) Krivično zakonodavstvo Srbije važi i za svakog ko učini krivično delo na domaćem brodu, bez obzira gde se brod nalazi u \
            vreme izvršenja dela. \
            (3) Krivično zakonodavstvo Srbije važi i za svakog ko učini krivično delo u domaćem civilnom vazduhoplovu dok je u letu ili u \
            domaćem vojnom vazduhoplovu, bez obzira gde se vazduhoplov nalazio u vreme izvršenja krivičnog dela. \
            (4) Ako je u slučajevima iz st. 1. do 3. ovog člana u stranoj državi pokrenut ili dovršen krivični postupak, krivično gonjenje u \
            Srbiji preduzeće se samo po odobrenju republičkog javnog tužioca. \
            (5) Krivično gonjenje stranca u slučajevima iz st. 1. do 3. ovog člana može se, pod uslovom uzajamnosti, ustupiti stranoj \
            državi.",

            "Odgovornost pravnih lica za krivična dela \
            Član 12 \
            Odgovornost pravnih lica za krivična dela, kao i sankcije pravnih lica za krivična dela uređuju se posebnim zakonom.",

            "Početak izvršenja kazne i obaveštavanje suda \
            Član 18 \
            Posebno odeljenje obaveštava predsednika suda, odnosno ovlašćenog sudiju, javnog tužioca i policiju o danu i času kada je \
            osuđeni stupio na izdržavanje kazne. \
            Početak izdržavanja kazne računa se od dana kada je osuđeni stupio na izdržavanje kazne u Posebno odeljenje.",

            "Tipovi pojedinih vrsta zavoda \
            Član 15 \
            Kazneno-popravni zavod za žene, okružni zatvor i vaspitno-popravni dom su poluotvorenog tipa. \
            Specijalna zatvorska bolnica i kazneno-popravni zavod za maloletnike su zatvorenog tipa.",

            "Dostavljanje odluke javnog tužilaštva \
            Član 12 \
            Javni tužilac dostavlja odluku o odlaganju krivičnog gonjenja sa podacima o ličnosti osumnjičenog nadležnoj povereničkoj \
            kancelariji u roku od tri dana od dana donošenja odluke."
            ]

# Importing test questions
with open("test_questions.json", "r") as infile:
    questions_data = json.load(infile)

questions = questions_data["questions"]
real_answers = questions_data["real_answers"]

model_qa = build_model(configs.squad.qa_multisberquad_bert, download=True)

# Recording results
results = pd.DataFrame(columns=["question",
                                "obtained answer",
                                "real answer",
                                "confidence"])

for i, question in enumerate(questions):
    result = model_qa([ARTICLES[i]], [question])

    obtained_answer = result[0][0]
    confidence = result[2][0]

    print(DOUBLE_LINE)
    print(f" {i+1}. | QUESTION: {question}")
    print(DOUBLE_LINE)

    print(LINE)
    print(f"Obtained answer: {obtained_answer} | Real answer: {real_answers[i]}")
    print(LINE)

    print(f"Confidence: {round(confidence, 2)}")
    print(LINE)

    data = {"question": question,
            "obtained answer": obtained_answer,
            "real answer": real_answers[i],
            "confidence": confidence}

    results = pd.concat([results, pd.DataFrame([data])], ignore_index=True)

# Results saving
results.to_csv("secondary qa test - results.csv", index=False)





