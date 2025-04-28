from django.shortcuts import render
from sentiment import utils


def index(request):
    content: str = request.POST.get('text')
    render_material = {"result":""}
    if content:
        tokens: list = utils.text_tokenize(content)
        render_material["result"]  = utils.analyze_sentiment(tokens)

    return render(request, 'index.html', render_material)