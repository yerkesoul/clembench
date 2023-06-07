# The Lightweight Dialogue Game framework

## How to add a new model as a backend

The backend is responsible for calling local or remote models (via an API). You can easily extend clembench with your own models.

1. Add a file that ends in `_api.py` in the backends directory e.g. `mybackend_api.py`
2. Implement in that file your backend class which needs to extend `backends.Backend` e.g. `class MyBackend(backends.Backend)`
3. Add an entry for your backend in the `key.json`

The framework will automatically look into the backends folder for all files that end in `_api.py`
and load all classes in these modules that extend `backends.Backend`.
