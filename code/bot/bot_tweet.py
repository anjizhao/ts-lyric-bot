
import os

from dotenv import load_dotenv
from twitter import OAuth, Twitter

load_dotenv()

TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')

TWITTER_BOT_OAUTH_TOKEN = os.getenv('TWITTER_BOT_OAUTH_TOKEN')
TWITTER_BOT_OAUTH_SECRET = os.getenv('TWITTER_BOT_OAUTH_SECRET')



t = Twitter(
    auth=OAuth(
        TWITTER_BOT_OAUTH_TOKEN,
        TWITTER_BOT_OAUTH_SECRET,
        TWITTER_CONSUMER_KEY,
        TWITTER_CONSUMER_SECRET,
    )
)


def post_tweet(text: str) -> None:
    t.statuses.update(status=text)

