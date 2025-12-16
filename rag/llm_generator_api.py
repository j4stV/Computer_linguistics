"""Альтернативный модуль для работы с LLM через API."""

from typing import List, Optional
import os


class LLMGeneratorAPI:
    """Класс для генерации ответов с помощью LLM через API."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """Инициализация генератора через API.
        
        Args:
            provider: Провайдер API ('openai', 'yandex', 'anthropic')
            api_key: API ключ (если None, берется из переменных окружения)
            model: Имя модели (если None, используется модель по умолчанию)
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.model = model or self._get_default_model()
        
        if not self.api_key:
            raise ValueError(f"API ключ не найден. Установите {provider.upper()}_API_KEY или передайте api_key")
        
        self._init_client()
    
    def _get_default_model(self) -> str:
        """Возвращает модель по умолчанию для провайдера."""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "yandex": "yandexgpt",
            "anthropic": "claude-3-haiku-20240307"
        }
        return defaults.get(self.provider, "gpt-3.5-turbo")
    
    def _init_client(self) -> None:
        """Инициализирует клиент API."""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Для использования OpenAI API установите: pip install openai")
        
        elif self.provider == "yandex":
            try:
                import yandexcloud
                from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import CompletionOptions
                from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import FoundationModelsServiceStub
                # Инициализация YandexGPT клиента
                # Требует дополнительной настройки
                self.client = None  # Заглушка
                print("YandexGPT требует дополнительной настройки")
            except ImportError:
                raise ImportError("Для использования YandexGPT установите: pip install yandexcloud")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Для использования Anthropic API установите: pip install anthropic")
        
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {self.provider}")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Генерирует ответ на основе промпта.
        
        Args:
            prompt: Текст промпта
            max_tokens: Максимальное количество токенов
            temperature: Температура для генерации
            
        Returns:
            Сгенерированный текст
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, max_tokens, temperature)
        else:
            raise NotImplementedError(f"Генерация для {self.provider} не реализована")
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Генерация через OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Генерация через Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def answer_question(self, question: str, context_texts: List[str]) -> str:
        """Генерирует ответ на вопрос на основе контекста.
        
        Args:
            question: Вопрос пользователя
            context_texts: Список текстовых фрагментов с контекстом
            
        Returns:
            Сгенерированный ответ
        """
        # Формируем промпт
        context = "\n\n".join(context_texts)
        prompt = f"""Дай ответ на данный вопрос, используя информацию из текста:
{question}

Текст:
{context}"""
        
        return self.generate(prompt, max_tokens=512, temperature=0.7)


