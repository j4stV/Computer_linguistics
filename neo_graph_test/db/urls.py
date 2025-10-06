from django.urls import path
from django.conf.urls import url

from db.views import (
    getTest,
    postTest,
    deleteTest,
    get_corpus,
    post_corpus,
    delete_corpus,
    get_text,
    post_text,
    delete_text,
    ontology_get,
    ontology_get_class,
    ontology_create_class,
)

urlpatterns = [
    path('getTest',getTest , name='getTest'),
    path('postTest',postTest , name='postTest'),
    path('deleteTest',deleteTest , name='deleteTest'),
    path('corpus/get', get_corpus, name='get_corpus'),
    path('corpus/post', post_corpus, name='post_corpus'),
    path('corpus/delete', delete_corpus, name='delete_corpus'),
    path('text/get', get_text, name='get_text'),
    path('text/post', post_text, name='post_text'),
    path('text/delete', delete_text, name='delete_text'),
    path('ontology/get', ontology_get, name='ontology_get'),
    path('ontology/get_class', ontology_get_class, name='ontology_get_class'),
    path('ontology/create_class', ontology_create_class, name='ontology_create_class'),
]