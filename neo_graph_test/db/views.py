from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseRedirect, HttpResponse
from django.forms.models import model_to_dict
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import datetime
from django.db.models import Q
from.onthology_namespace import *
from .models import Test, Corpus, Text
from core.settings import *

# API IMPORTS
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny

# REPO IMPORTS
from db.api.TestRepository import TestRepository, CorpusRepository, TextRepository
from db.api.OntologyRepository import OntologyRepository

@api_view(['GET', ])
@permission_classes((AllowAny,))
def getTest(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    
    testRepo = TestRepository()
    result = testRepo.getTest(id = id)
    return Response(result)

@api_view(['POST', ])
@permission_classes((IsAuthenticated,))
def postTest(request):
    data = json.loads(request.body.decode('utf-8'))
    testRepo = TestRepository()
    test = testRepo.postTest(test_data = data)
    return JsonResponse(test)

@api_view(['DELETE', ])
@permission_classes((AllowAny,))
def deleteTest(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    
    testRepo = TestRepository()
    result = testRepo.deleteTest(id = id)
    return Response(result)


# -------------------- Corpus CRUD --------------------
@api_view(['GET', ])
@permission_classes((AllowAny,))
def get_corpus(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    repo = CorpusRepository()
    result = repo.get_corpus(id=int(id))
    return Response(result)


@api_view(['POST', ])
@permission_classes((IsAuthenticated,))
def post_corpus(request):
    data = json.loads(request.body.decode('utf-8'))
    repo = CorpusRepository()
    corpus = repo.create_or_update_corpus(corpus_data=data)
    return JsonResponse(corpus)


@api_view(['DELETE', ])
@permission_classes((AllowAny,))
def delete_corpus(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    repo = CorpusRepository()
    result = repo.delete_corpus(id=int(id))
    return Response(result)


# -------------------- Text CRUD --------------------
@api_view(['GET', ])
@permission_classes((AllowAny,))
def get_text(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    repo = TextRepository()
    result = repo.get_text(id=int(id))
    return Response(result)


@api_view(['POST', ])
@permission_classes((IsAuthenticated,))
def post_text(request):
    data = json.loads(request.body.decode('utf-8'))
    repo = TextRepository()
    text = repo.create_or_update_text(text_data=data)
    return JsonResponse(text)


@api_view(['DELETE', ])
@permission_classes((AllowAny,))
def delete_text(request):
    id = request.GET.get('id', None)
    if id is None:
        return HttpResponse(status=400)
    repo = TextRepository()
    result = repo.delete_text(id=int(id))
    return Response(result)


# -------------------- OntologyRepository endpoints (examples) --------------------
@api_view(['GET', ])
@permission_classes((AllowAny,))
def ontology_get(request):
    # В реальном проекте возьмите из settings переменные подключения
    uri = request.GET.get('uri', 'bolt://localhost:7687')
    user = request.GET.get('user', 'neo4j')
    password = request.GET.get('password', 'password')
    repo = OntologyRepository(uri=uri, user=user, password=password)
    try:
        data = repo.get_ontology()
        return Response(data)
    finally:
        repo.close()


@api_view(['GET', ])
@permission_classes((AllowAny,))
def ontology_get_class(request):
    class_uri = request.GET.get('class_uri')
    if not class_uri:
        return HttpResponse(status=400)
    uri = request.GET.get('uri', 'bolt://localhost:7687')
    user = request.GET.get('user', 'neo4j')
    password = request.GET.get('password', 'password')
    repo = OntologyRepository(uri=uri, user=user, password=password)
    try:
        data = repo.get_class(class_uri)
        return Response(data)
    finally:
        repo.close()


@api_view(['POST', ])
@permission_classes((IsAuthenticated,))
def ontology_create_class(request):
    body = json.loads(request.body.decode('utf-8'))
    title = body.get('title', '')
    description = body.get('description', '')
    parent_uri = body.get('parent_uri')

    uri = body.get('uri', 'bolt://localhost:7687')
    user = body.get('user', 'neo4j')
    password = body.get('password', 'password')

    repo = OntologyRepository(uri=uri, user=user, password=password)
    try:
        new_uri = repo.create_class(title=title, description=description, parent_uri=parent_uri)
        return Response({'uri': new_uri})
    finally:
        repo.close()