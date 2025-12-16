"""Простой тестовый скрипт для проверки работы RAG системы."""

from rag.rag_system import RAGSystem


def test_basic_functionality():
    """Тестирует базовую функциональность системы."""
    
    ontology_files = [
        r"c:\Users\just_\Downloads\graph(5).json",
        r"c:\Users\just_\Downloads\graph (2).json"
    ]
    
    print("Тест 1: Инициализация системы...")
    try:
        rag_system = RAGSystem(
            ontology_files=ontology_files,
            llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
            cache_dir="./rag_cache",
            n_nodes=5,  # Меньше узлов для быстрого теста
            m_nodes=3
        )
        print("✓ Система инициализирована успешно\n")
    except Exception as e:
        print(f"✗ Ошибка инициализации: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    print("Тест 2: Выполнение простого запроса...")
    try:
        question = "Над какими проектами работал Серый А.С.?"
        answer = rag_system.query(question, verbose=False)
        print(f"Вопрос: {question}")
        print(f"Ответ: {answer[:200]}...\n")
        print("✓ Запрос выполнен успешно\n")
    except Exception as e:
        print(f"✗ Ошибка выполнения запроса: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functionality()


