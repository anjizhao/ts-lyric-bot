
import os

from dotenv import load_dotenv
from twitter import oauth_dance


load_dotenv()

TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')

if __name__ == '__main__':
    oauth_dance(
        'anjis-ts-bot',
        TWITTER_CONSUMER_KEY,
        TWITTER_CONSUMER_SECRET,
        token_filename='.bot_account_oauth',
    )

