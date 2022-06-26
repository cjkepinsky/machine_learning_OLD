# Uncomment the following lines to enable verbose logging
# import logging
# logging.basicConfig(level=logging.INFO)

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
print("Oeezu, czego tam? Zara zejdę....")

logic_adapters = [
    {
        "chatterbot.logic.BestMatch"
    },
    {
        "import_path": "logic.news",
        'input_text': 'dawaj newsy'
    }
]
bot = ChatBot('Chatbot', logic_adapters=logic_adapters)
print('No, no! Nie tupie tam, tylko czeka!')
trainer = ChatterBotCorpusTrainer(bot)
trainer.train("polish")

print('Zdzisław Skośnooki jestem, czekam aż sie ockniesz...')
# print("No jestem, co tam?")

# The following loop will execute each time the user enters input
while True:
    try:
        user_input = input()
        bot_response = bot.get_response(user_input)
        print(bot_response)
    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break