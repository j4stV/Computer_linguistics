"""Пример использования RAG системы для работы с онтологиями."""

from rag.rag_system import RAGSystem


def main():
    """Основная функция для демонстрации работы RAG системы."""
    
    # Пути к файлам онтологий
    ontology_files = [
        r"c:\Users\just_\Downloads\graph(5).json",
        r"c:\Users\just_\Downloads\graph (2).json"
    ]
    
    # Инициализация RAG системы
    print("="*60)
    print("Инициализация RAG системы")
    print("="*60)
    
    rag_system = RAGSystem(
        ontology_files=ontology_files,
        llm_model_name="meta-llama/Llama-3.1-8B-Instruct",  # Можно изменить на другую модель
        cache_dir="./rag_cache",  # Директория для кэширования эмбеддингов
        n_nodes=10,  # Количество узлов для первого поиска
        m_nodes=5    # Количество узлов для второго поиска
    )
    
    # Примеры запросов
    questions = [
        "Какие гипотезы предложил Петров?",
        "Хорошая ли гипотеза с персонализированными предложениями?",
        "Какие эксперименты имели успешный результат?",
    ]
    
    # Выполнение запросов
    print("\n" + "="*60)
    print("Выполнение запросов")
    print("="*60 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'#'*60}")
        print(f"Запрос {i}/{len(questions)}")
        print(f"{'#'*60}\n")
        
        try:
            answer = rag_system.query(question, verbose=True)
            print(f"\nОтвет: {answer}\n")
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


