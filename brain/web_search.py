"""
Recherche web pour Atlas.
Interroge DuckDuckGo Instant Answer API (sans clé) puis Wikipedia
afin de répondre à des questions non couvertes par le modèle local.
"""

import logging
import urllib.parse

import requests

logger = logging.getLogger("atlas.web_search")

# Timeout commun pour toutes les requêtes HTTP
_TIMEOUT = 6

# Longueur maximale d'un extrait Wikipedia
_MAX_EXTRACT_LENGTH = 600


def _duckduckgo(question: str) -> str | None:
    """
    Interroge l'API DuckDuckGo Instant Answer.
    Retourne la réponse textuelle la plus pertinente ou None.
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": question,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "no_redirect": "1",
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("DuckDuckGo indisponible : %s", exc)
        return None

    # Priorité 1 : réponse directe (calcul, conversion, etc.)
    if data.get("Answer"):
        return str(data["Answer"])

    # Priorité 2 : résumé abstrait (article encyclopédique)
    if data.get("AbstractText"):
        source = data.get("AbstractURL", "")
        text = data["AbstractText"]
        if source:
            return f"{text}\n🔗 Source : {source}"
        return text

    # Priorité 3 : premier sujet lié
    topics = data.get("RelatedTopics", [])
    for topic in topics:
        # Les topics peuvent être des groupes (avec sous-clé "Topics")
        if isinstance(topic, dict) and topic.get("Text"):
            url_topic = topic.get("FirstURL", "")
            text = topic["Text"]
            if url_topic:
                return f"{text}\n🔗 En savoir plus : {url_topic}"
            return text

    return None


def _wikipedia(question: str) -> str | None:
    """
    Recherche un résumé Wikipedia pour la question.
    Retourne un extrait + lien ou None.
    """
    # Étape 1 : recherche du titre le plus proche
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "srlimit": "1",
        "format": "json",
    }
    try:
        resp = requests.get(search_url, params=search_params, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]
    except Exception as exc:
        logger.warning("Wikipedia (search) indisponible : %s", exc)
        return None

    # Étape 2 : récupérer le résumé de la page trouvée
    summary_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    )
    try:
        resp = requests.get(summary_url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if extract:
            # Limiter à ~600 caractères pour rester lisible
            if len(extract) > _MAX_EXTRACT_LENGTH:
                extract = extract[:_MAX_EXTRACT_LENGTH].rstrip() + "…"
            if page_url:
                return f"{extract}\n🔗 Wikipedia : {page_url}"
            return extract
    except Exception as exc:
        logger.warning("Wikipedia (summary) indisponible : %s", exc)

    return None


def _wikipedia_fr(question: str) -> str | None:
    """
    Même chose mais sur Wikipedia francophone.
    """
    search_url = "https://fr.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "srlimit": "1",
        "format": "json",
    }
    try:
        resp = requests.get(search_url, params=search_params, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]
    except Exception as exc:
        logger.warning("Wikipedia FR (search) indisponible : %s", exc)
        return None

    summary_url = (
        f"https://fr.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    )
    try:
        resp = requests.get(summary_url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if extract:
            if len(extract) > _MAX_EXTRACT_LENGTH:
                extract = extract[:_MAX_EXTRACT_LENGTH].rstrip() + "…"
            if page_url:
                return f"{extract}\n🔗 Wikipedia : {page_url}"
            return extract
    except Exception as exc:
        logger.warning("Wikipedia FR (summary) indisponible : %s", exc)

    return None


def chercher(question: str) -> str | None:
    """
    Recherche la meilleure réponse sur internet pour la question donnée.
    Tente dans l'ordre :
      1. DuckDuckGo Instant Answer
      2. Wikipedia français
      3. Wikipedia anglais
    Retourne None si aucune source ne répond.
    """
    reponse = _duckduckgo(question)
    if reponse:
        return reponse

    reponse = _wikipedia_fr(question)
    if reponse:
        return reponse

    reponse = _wikipedia(question)
    return reponse
