from db.models import Test, Corpus, Text

class TestRepository:
    def __init__(self, ):
        pass

    def collect_test(self, test: Test):
        temp = {
            'id': test.pk,
            'name': test.name
        }
        return temp

    def getTest(self, id):
        test = Test.objects.get(pk = id)
        return self.collect_test(test)
    
    # Can update test, if "id" in test_data
    def postTest(self, test_data): 
        if 'id' in test_data:
            test = Test.objects.get(pk = id)
        else:
            test = Test()

        test.name = test_data.get('name', '')
        test.save()
        return self.collect_test(test)
    
    def deleteTest(self, id):
        test = Test.objects.get(pk = id)
        test.delete()
        return id


class CorpusRepository:
    def __init__(self):
        pass

    def collect_corpus(self, corpus: Corpus, include_texts: bool = True):
        data = {
            'id': corpus.pk,
            'title': corpus.title,
            'description': corpus.description,
            'genre': corpus.genre,
        }
        if include_texts:
            data['texts'] = [
                {
                    'id': t.pk,
                    'title': t.title,
                    'description': t.description,
                    'content': t.content,
                    'has_translation': t.has_translation_id,
                }
                for t in corpus.texts.all()
            ]
        return data

    def get_corpus(self, id: int):
        corpus = Corpus.objects.get(pk=id)
        return self.collect_corpus(corpus)

    def create_or_update_corpus(self, corpus_data):
        corpus_id = corpus_data.get('id')
        if corpus_id:
            corpus = Corpus.objects.get(pk=corpus_id)
        else:
            corpus = Corpus()
        corpus.title = corpus_data.get('title', '')
        corpus.description = corpus_data.get('description', '')
        corpus.genre = corpus_data.get('genre', '')
        corpus.save()
        return self.collect_corpus(corpus, include_texts=False)

    def delete_corpus(self, id: int):
        corpus = Corpus.objects.get(pk=id)
        corpus.delete()
        return id


class TextRepository:
    def __init__(self):
        pass

    def collect_text(self, text: Text):
        return {
            'id': text.pk,
            'title': text.title,
            'description': text.description,
            'content': text.content,
            'corpus_id': text.corpus_id,
            'has_translation': text.has_translation_id,
        }

    def get_text(self, id: int):
        text = Text.objects.get(pk=id)
        return self.collect_text(text)

    def create_or_update_text(self, text_data):
        text_id = text_data.get('id')
        if text_id:
            text = Text.objects.get(pk=text_id)
        else:
            text = Text()

        text.title = text_data.get('title', '')
        text.description = text_data.get('description', '')
        text.content = text_data.get('content', '')

        corpus_id = text_data.get('corpus_id')
        if corpus_id is not None:
            text.corpus = Corpus.objects.get(pk=corpus_id)

        has_translation = text_data.get('has_translation')
        if has_translation is not None:
            if has_translation == '':
                text.has_translation = None
            else:
                text.has_translation = Text.objects.get(pk=has_translation)

        text.save()
        return self.collect_text(text)

    def delete_text(self, id: int):
        text = Text.objects.get(pk=id)
        text.delete()
        return id
