
from code.bot.bot_tweet import post_tweet
from code.ngrams import MyNGram

from code.save_utils import load_model


model_filename = 'data/saved_models/kn3_redtv.json'


def main():
    model = MyNGram(load_model(model_filename))
    text = None
    while not text:
        text = model.generate_sentence(40)
    post_tweet(text)


if __name__ == '__main__':
    main()





# with open("data/outputs.txt", "a") as openfile:
#     for i in range(40):
#             openfile.write(model.generate_sentence(50)+"\n")
