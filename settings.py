import dj_database_url
import django_heroku
import os


DEBUG = False
# remove https://from the url
# use * if you want to allow all hosts.
ALLOWED_HOSTS = ['<url>']

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql_psycopg2',
            'NAME': 'ciba',
        }
    }

STATIC_ROOT = os.path.join("EclipsingBinaries", 'staticfiles')
STATIC_URL = '/static/'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

django_heroku.settings(locals())
