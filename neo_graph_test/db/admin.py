from django.contrib import admin

from django.db import models
from .models import Test


class TestAdmin(admin.ModelAdmin):
    model = Test

admin.site.register(Test, TestAdmin)