from django.apps import AppConfig

class SentimentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sentiment"
    _initialized = False

    def ready(self):
        if not SentimentConfig._initialized:
            SentimentConfig._initialized = True
            self.run_initialization()

    def run_initialization(self):
        print("Running initialization")
        from sentiment.utils.sentiment_processing import train_model
        train_model()