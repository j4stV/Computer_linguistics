"""Модуль для работы с LLM (генерация ответов)."""

from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMGenerator:
    """Класс для генерации ответов с помощью LLM."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: Optional[str] = None, use_8bit: bool = False):
        """Инициализация генератора.
        
        Args:
            model_name: Имя модели из HuggingFace
            device: Устройство для вычислений ('cuda', 'cpu' или None для автоопределения)
            use_8bit: Использовать ли 8-bit квантование для экономии памяти (быстрее для маленьких моделей)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_8bit = use_8bit
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.model_device = self.device  # Устройство модели (может отличаться от self.device при 8-bit)
        self.max_length = 256  # По умолчанию меньше для быстрой генерации
        self.temperature = 0.3  # По умолчанию меньше для быстрой генерации
        self._load_model()
    
    def _load_model(self) -> None:
        """Загружает модель и токенизатор."""
        print(f"Загрузка модели {self.model_name} на устройство {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Настройки загрузки для оптимизации
            load_kwargs = {}
            
            # 8-bit квантование для экономии памяти и ускорения (только для CUDA)
            if self.use_8bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    # Проверяем наличие accelerate перед использованием bitsandbytes
                    try:
                        import accelerate
                        from packaging import version
                        # Проверяем версию accelerate
                        accel_version = getattr(accelerate, '__version__', '0.0.0')
                        if version.parse(accel_version) < version.parse('0.26.0'):
                            raise ImportError(f"accelerate version {accel_version} too old, need >= 0.26.0")
                    except ImportError as e:
                        print(f"accelerate не установлен или версия недостаточна: {e}")
                        print("Установите: pip install 'accelerate>=0.26.0'")
                        raise ImportError("accelerate required for 8-bit quantization")
                    
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    print("Используется 8-bit квантование")
                except (ImportError, Exception) as e:
                    print(f"8-bit квантование недоступно ({e}), используем обычную загрузку")
                    self.use_8bit = False  # Отключаем 8-bit для дальнейшей загрузки
            
            # Тип данных и размещение устройства
            if self.device == "cuda" and not self.use_8bit:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Определяем фактическое устройство модели
            if self.use_8bit and self.device == "cuda":
                # При 8-bit квантовании модель автоматически на CUDA
                # Определяем конкретное устройство (может быть cuda:0, cuda:1 и т.д.)
                if hasattr(self.model, 'device'):
                    self.model_device = str(self.model.device)
                elif hasattr(self.model, 'hf_device_map'):
                    # Если используется device_map, берем первое устройство
                    device_map = self.model.hf_device_map
                    if device_map:
                        first_device = list(device_map.values())[0]
                        self.model_device = str(first_device) if isinstance(first_device, torch.device) else "cuda"
                    else:
                        self.model_device = "cuda"
                else:
                    self.model_device = "cuda"
            elif self.device == "cpu" and not self.use_8bit:
                self.model = self.model.to(self.device)
                self.model_device = "cpu"
            elif self.device == "cuda" and not self.use_8bit:
                # Модель уже на CUDA через device_map="auto"
                if hasattr(self.model, 'device'):
                    self.model_device = str(self.model.device)
                else:
                    self.model_device = "cuda"
            else:
                self.model_device = self.device
            
            # Устанавливаем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Модель загружена успешно")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Используется заглушка для генерации ответов")
            self.tokenizer = None
            self.model = None
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Генерирует ответ на основе промпта.
        
        Args:
            prompt: Текст промпта
            max_length: Максимальная длина генерируемого текста
            temperature: Температура для генерации (контролирует случайность)
            
        Returns:
            Сгенерированный текст
        """
        if self.model is None or self.tokenizer is None:
            # Заглушка для случая, когда модель не загружена
            return self._generate_stub(prompt)
        
        try:
            # Формируем промпт в формате модели
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Форматируем промпт для разных типов моделей
            if "llama" in self.model_name.lower() or "instruct" in self.model_name.lower():
                try:
                    # Пытаемся использовать chat template
                    if hasattr(self.tokenizer, "apply_chat_template"):
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        formatted_prompt = prompt
                except Exception:
                    # Если не получается, используем простой промпт
                    formatted_prompt = prompt
            else:
                formatted_prompt = prompt
            
            # Токенизируем
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
            # Перемещаем входные данные на то же устройство, где находится модель
            model_device = getattr(self, 'model_device', self.device)
            inputs = inputs.to(model_device)
            
            # Генерируем с оптимизацией для скорости
            with torch.no_grad():
                # Для маленьких моделей используем greedy decoding если temperature очень низкая
                do_sample = temperature > 0.1
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    num_beams=1 if do_sample else 1,  # Без beam search для скорости
                )
            
            # Декодируем ответ
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем промпт из ответа
            if formatted_prompt in generated_text:
                generated_text = generated_text.replace(formatted_prompt, "").strip()
            
            return generated_text
        except Exception as e:
            print(f"Ошибка при генерации: {e}")
            return self._generate_stub(prompt)
    
    def _generate_stub(self, prompt: str) -> str:
        """Заглушка для генерации ответов без модели.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            Простой ответ-заглушка
        """
        # Простая заглушка, которая извлекает информацию из промпта
        if "Текст:" in prompt:
            text_part = prompt.split("Текст:")[-1]
            return f"На основе предоставленной информации: {text_part[:200]}..."
        return "Ответ сгенерирован на основе предоставленной информации."
    
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
        
        return self.generate(prompt, max_length=self.max_length, temperature=self.temperature)


