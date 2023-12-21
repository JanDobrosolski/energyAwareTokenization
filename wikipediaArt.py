import wikipediaapi
import wikipedia

NUM_ARTICLES = 2

def get_article(lang1, lang2):
    wikipedia.set_lang(lang1)
    while True:
        try:
            title = wikipedia.random(1)
            pageLang1 = wikipedia.page(title)
            contentLang1 = pageLang1.content
        except wikipedia.exceptions.DisambiguationError as e:
            continue
        except wikipedia.exceptions.PageError:
            continue


        wikipedia.set_lang(lang2)
        try:
            pageLang2 = wikipedia.page(title)
            contentLang2 = pageLang2.content
            return contentLang1, contentLang2
        except wikipedia.exceptions.PageError:
            continue


def save_to_file(fileName, text):
    if text:
        with open(fileName, 'a', encoding='utf-8') as file:
            file.write(text)


for i in range(NUM_ARTICLES):
    english_article, polish_article = get_article('en', 'pl')
    import pdb; pdb.set_trace();

    save_to_file('en_articles.txt', english_article)
    save_to_file('pl_articles.txt', polish_article)


print("Articles saved")


