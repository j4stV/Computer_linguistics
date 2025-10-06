from django.db import models
from db_file_storage.model_utils import delete_file, delete_file_if_needed


class Test(models.Model):
    name = models.TextField()

    def __str__(self):
        return self.name  # Returns the value of the 'name' field
    

class Corpus(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    genre = models.CharField(max_length=128, blank=True, default='')

    def __str__(self):
        return self.title


class Text(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    content = models.TextField()
    corpus = models.ForeignKey(Corpus, related_name='texts', on_delete=models.CASCADE)
    # связь с самим собой (опциональная) — указывает на текст-перевод или оригинал
    has_translation = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='translations')

    def __str__(self):
        return self.title
