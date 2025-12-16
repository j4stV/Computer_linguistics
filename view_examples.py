"""Скрипт для просмотра примеров текстов, сгенерированных из узлов онтологии."""

from rag.ontology_loader import OntologyLoader
from rag.text_transformer import TextTransformer
import json

# Настройки (измените под свои файлы)
ONTOLOGY_FILES = [
    r"c:\Users\just_\Downloads\graph(5).json",
    r"c:\Users\just_\Downloads\graph (2).json"
]

def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Печатает разделитель с заголовком."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)

def view_node_examples():
    """Просмотр примеров текстов узлов."""
    
    print_separator("ПРОСМОТР ПРИМЕРОВ ТЕКСТОВ ИЗ ОНТОЛОГИИ")
    
    # Загрузка онтологий
    print("\n📂 Загрузка онтологий...")
    loader = OntologyLoader()
    loader.load_multiple_files(ONTOLOGY_FILES)
    
    nodes = loader.get_all_nodes()
    edges = loader.get_all_edges()
    
    print(f"✅ Загружено узлов: {len(nodes)}")
    print(f"✅ Загружено связей: {len(edges)}\n")
    
    # Инициализация трансформера
    transformer = TextTransformer(loader)
    
    # Преобразование всех узлов в тексты
    print("🔄 Преобразование узлов в тексты...")
    node_texts = transformer.transform_all_nodes()
    
    print(f"✅ Сгенерировано текстов: {len(node_texts)}\n")
    
    # Статистика по типам узлов
    print_separator("СТАТИСТИКА ПО ТИПАМ УЗЛОВ")
    
    type_counts = {}
    for node in nodes:
        node_data = node.get('data', {})
        labels = node_data.get('labels', [])
        
        node_type = None
        if 'http://www.w3.org/2002/07/owl#Class' in labels:
            node_type = 'Class'
        elif 'http://www.w3.org/2002/07/owl#NamedIndividual' in labels:
            node_type = 'Object'
        elif 'http://www.w3.org/2002/07/owl#DatatypeProperty' in labels:
            node_type = 'DatatypeProperty'
        elif 'http://www.w3.org/2002/07/owl#ObjectProperty' in labels:
            node_type = 'ObjectProperty'
        else:
            node_type = 'Unknown'
        
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type}: {count}")
    
    # Показываем примеры текстов
    print_separator("ПРИМЕРЫ ТЕКСТОВ УЗЛОВ")
    
    # Показываем первые 5 примеров
    print(f"\n📝 Показываем первые 5 примеров из {len(node_texts)}:\n")
    
    for i, text in enumerate(node_texts[:5], 1):
        print_separator(f"Пример {i}", char="-", width=60)
        print(text)
        print(f"\nДлина текста: {len(text)} символов")
        print(f"Количество строк: {text.count(chr(10)) + 1}")
    
    # Показываем примеры разных типов узлов
    print_separator("ПРИМЕРЫ ПО ТИПАМ УЗЛОВ")
    
    examples_by_type = {}
    for i, node in enumerate(nodes):
        node_data = node.get('data', {})
        labels = node_data.get('labels', [])
        
        node_type = None
        if 'http://www.w3.org/2002/07/owl#Class' in labels:
            node_type = 'Class'
        elif 'http://www.w3.org/2002/07/owl#NamedIndividual' in labels:
            node_type = 'Object'
        elif 'http://www.w3.org/2002/07/owl#DatatypeProperty' in labels:
            node_type = 'DatatypeProperty'
        elif 'http://www.w3.org/2002/07/owl#ObjectProperty' in labels:
            node_type = 'ObjectProperty'
        else:
            node_type = 'Unknown'
        
        if node_type not in examples_by_type:
            examples_by_type[node_type] = []
        
        if len(examples_by_type[node_type]) < 2:  # По 2 примера каждого типа
            text = transformer.node_to_text(node)
            if text.strip():
                examples_by_type[node_type].append(text)
    
    for node_type, examples in sorted(examples_by_type.items()):
        if examples:
            print_separator(f"Тип: {node_type}", char="-", width=60)
            for j, example in enumerate(examples, 1):
                print(f"\nПример {j}:")
                print(example[:500] + ("..." if len(example) > 500 else ""))
                print()
    
    # Показываем самые длинные и короткие тексты
    print_separator("СТАТИСТИКА ПО ДЛИНЕ ТЕКСТОВ")
    
    text_lengths = [(len(text), i, text[:100]) for i, text in enumerate(node_texts)]
    text_lengths.sort()
    
    print("\n📏 Самые короткие тексты:")
    for length, idx, preview in text_lengths[:3]:
        print(f"  Узел {idx}: {length} символов - {preview}...")
    
    print("\n📏 Самые длинные тексты:")
    for length, idx, preview in text_lengths[-3:]:
        print(f"  Узел {idx}: {length} символов - {preview}...")
    
    # Средняя длина
    avg_length = sum(len(text) for text in node_texts) / len(node_texts) if node_texts else 0
    print(f"\n📊 Средняя длина текста: {avg_length:.1f} символов")
    
    # Интерактивный просмотр
    print_separator("ИНТЕРАКТИВНЫЙ ПРОСМОТР")
    print("\nКоманды:")
    print("  - Введите номер узла (0-{}) для просмотра его текста".format(len(node_texts) - 1))
    print("  - Введите 's <текст>' для поиска узлов по тексту")
    print("  - Введите 'q' для выхода")
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if user_input.lower() == 'q':
                break
            
            # Поиск по тексту
            if user_input.lower().startswith('s '):
                search_term = user_input[2:].strip().lower()
                if not search_term:
                    print("❌ Введите текст для поиска после 's '")
                    continue
                
                print(f"\n🔍 Поиск по тексту: '{search_term}'")
                found_count = 0
                for idx, text in enumerate(node_texts):
                    if search_term in text.lower():
                        found_count += 1
                        if found_count <= 5:  # Показываем первые 5 результатов
                            print_separator(f"Найден узел {idx}", char="-", width=60)
                            print(text[:300] + ("..." if len(text) > 300 else ""))
                
                if found_count == 0:
                    print("❌ Ничего не найдено")
                elif found_count > 5:
                    print(f"\n... и еще {found_count - 5} результатов. Уточните поиск.")
                else:
                    print(f"\n✅ Найдено результатов: {found_count}")
                continue
            
            # Просмотр по номеру
            node_idx = int(user_input)
            if 0 <= node_idx < len(node_texts):
                print_separator(f"Узел {node_idx}", char="-", width=60)
                print(node_texts[node_idx])
                
                # Показываем исходные данные узла
                print("\n📋 Исходные данные узла (JSON, первые 500 символов):")
                node = nodes[node_idx]
                node_json = json.dumps(node, ensure_ascii=False, indent=2)
                print(node_json[:500] + ("..." if len(node_json) > 500 else ""))
            else:
                print(f"❌ Номер должен быть от 0 до {len(node_texts) - 1}")
        except ValueError:
            print("❌ Введите число, 's <текст>' для поиска или 'q' для выхода")
        except KeyboardInterrupt:
            print("\n\n👋 Выход...")
            break
    
    print_separator("КОНЕЦ ПРОСМОТРА")


if __name__ == "__main__":
    try:
        view_node_examples()
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Файл не найден - {e}")
        print("Проверьте пути к файлам онтологий в ONTOLOGY_FILES")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

