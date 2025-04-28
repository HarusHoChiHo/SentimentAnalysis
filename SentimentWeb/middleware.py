class InitializationMiddleware(object):
    initialized = False

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not self.initialized:

            self.initialized = True

        response = self.get_response(request)
        return response