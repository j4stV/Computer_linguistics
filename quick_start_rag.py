"""Быстрый запуск RAG системы - минимальный код для старта с Mistral API."""

from rag.rag_system import RAGSystem
import os

# Настройки (измените под свои файлы)
ONTOLOGY_FILES = [
    "graph.json",  # Используем локальный файл
]

# ИСПОЛЬЗОВАНИЕ MISTRAL API:
# 1. Установите переменную окружения MISTRAL_API_KEY или передайте api_key
# 2. Доступные модели: "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"
MISTRAL_MODEL = "mistral-small-latest"

# 2. Минимальное количество узлов для поиска
N_NODES = 3  # Первый поиск
M_NODES = 2  # Второй поиск

print(f" Модель: {MISTRAL_MODEL}")
print(f"  Параметры поиска: N={N_NODES}, M={M_NODES}\n")

# Проверка наличия API ключа
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("  ВНИМАНИЕ: MISTRAL_API_KEY не найден в переменных окружения!")
    print("   Установите его командой: export MISTRAL_API_KEY='ваш_ключ'")
    print("   Или передайте api_key при инициализации RAGSystem\n")

# Быстрая инициализация с Mistral API
rag = RAGSystem(
    ontology_files=ONTOLOGY_FILES,
    llm_model_name=MISTRAL_MODEL,
    cache_dir="./rag_cache",  # Используем кэш для быстрой загрузки эмбеддингов
    n_nodes=N_NODES,
    m_nodes=M_NODES,
    use_api=True,  # Используем API вместо локальной модели
    api_provider="mistral",  # Провайдер API
    api_key=api_key  # Можно передать напрямую или использовать переменную окружения
)

# Оптимизируем генератор для быстрой работы (уже установлено по умолчанию)
# rag.llm_generator.max_length = 256  # Меньше токенов = быстрее
# rag.llm_generator.temperature = 0.3  # Меньше случайности = быстрее

# Задайте вопрос
question = "Кем запущен тест про изменение цвета кнопки?"
print(f"\n❓ Вопрос: {question}\n")

# Получите ответ
answer = rag.query(question, verbose=True)
print(f"Ответ: {answer}")

