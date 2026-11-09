"""HackerNews tech digest: summarize tech-sector news and email it with a podcast MP3."""

import base64
import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    chat_parse_to_model,
    get_mistral_client,
)
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail = connector("gmail")

HN_TOPSTORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
TTS_MODEL = "voxtral-mini-tts-3-0"
TTS_VOICE = "fr_marie_neutral"
SUMMARY_MODEL = "mistral-small-latest"


class HnStory(pydantic.BaseModel):
    id: int
    title: str
    url: str | None = None
    score: int = 0


class TechSelection(pydantic.BaseModel):
    stories: list[HnStory]


class Digest(pydantic.BaseModel):
    subject: str
    text_body: str
    html_body: str
    podcast_script: str


class DigestInput(pydantic.BaseModel):
    to: str = "alexi.loison@gmail.com"
    max_stories: int = 30
    dry_run: bool = False


class DigestOutput(pydantic.BaseModel):
    subject: str
    text_body: str
    html_body: str
    podcast_script: str
    selected_stories: list[HnStory]
    audio_bytes: int = 0
    sent: bool
    message_id: str | None = None


@workflows.activity(name="fetch-hn-front-page", retry_policy_max_attempts=3)
async def fetch_hn_front_page(max_stories: int) -> list[HnStory]:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(HN_TOPSTORIES_URL)
        response.raise_for_status()
        story_ids = response.json()[:max_stories]
        stories: list[HnStory] = []
        for story_id in story_ids:
            item = await client.get(HN_ITEM_URL.format(item_id=story_id))
            item.raise_for_status()
            data = item.json()
            if data.get("type") != "story" or not data.get("title"):
                continue
            stories.append(
                HnStory(
                    id=story_id,
                    title=data["title"],
                    url=data.get("url"),
                    score=data.get("score", 0),
                )
            )
        return stories


@workflows.activity(name="select-tech-stories", retry_policy_max_attempts=3)
async def select_tech_stories(stories: list[HnStory]) -> list[HnStory]:
    listing = "\n".join(
        f"- [{story.id}] {story.title} (score: {story.score}, url: {story.url or 'n/a'})"
        for story in stories
    )
    request = ChatCompletionRequest(
        model=SUMMARY_MODEL,
        messages=[
            UserMessage(
                content=(
                    "Voici des titres de la page d'accueil de HackerNews :\n\n"
                    f"{listing}\n\n"
                    "Sélectionne uniquement les actualités du secteur technologique : "
                    "entreprises tech, produits, IA, sécurité informatique, matériel, "
                    "open source, régulation technologique, science appliquée au numérique. "
                    "Exclus les blogs personnels, les essais d'opinion sans actualité, "
                    "et tout sujet hors du secteur tech. "
                    "Classe par importance décroissante, 10 titres maximum. "
                    "Réponds avec les identifiants HackerNews d'origine."
                )
            )
        ],
    )
    selection = await chat_parse_to_model(TechSelection, request)
    return selection.stories


@workflows.activity(name="build-digest", retry_policy_max_attempts=3)
async def build_digest(stories: list[HnStory]) -> Digest:
    listing = "\n".join(
        f"- {story.title} (score: {story.score}, url: {story.url or 'n/a'})"
        for story in stories
    )
    request = ChatCompletionRequest(
        model=SUMMARY_MODEL,
        messages=[
            UserMessage(
                content=(
                    "Voici les actualités tech sélectionnées de HackerNews :\n\n"
                    f"{listing}\n\n"
                    "Rédige en français :\n"
                    "1. `subject` : un objet d'email court et informatif.\n"
                    "2. `text_body` : pour chaque actualité, un titre suivi d'un résumé "
                    "de 2 à 3 phrases expliquant l'essentiel et pourquoi c'est important.\n"
                    "3. `html_body` : le même contenu en HTML simple et lisible "
                    "(titres en gras, listes, liens vers les sources).\n"
                    "4. `podcast_script` : un script de podcast fluide de 220 à 260 mots, "
                    "à lire à voix haute, qui présente les titres principaux avec une "
                    "phrase de contexte chacun, une courte introduction et une conclusion. "
                    "Texte parlé uniquement, sans balise ni indication de mise en scène."
                )
            )
        ],
    )
    return await chat_parse_to_model(Digest, request)


@workflows.activity(name="generate-podcast-audio", retry_policy_max_attempts=2)
async def generate_podcast_audio(script: str) -> str:
    client = get_mistral_client()
    response = client.audio.speech.complete(
        model=TTS_MODEL,
        input=script,
        voice_id=TTS_VOICE,
        response_format="mp3",
    )
    return response.audio_data


@workflows.activity(name="send-digest-email", retry_policy_max_attempts=1)
async def send_digest_email(
    to: str,
    subject: str,
    text_body: str,
    html_body: str,
    audio_base64: str,
    client: ToolCallClient = Depends(gmail),
) -> str:
    result = await client.call_tool(
        tool_name="send_email",
        arguments={
            "to": [to],
            "subject": subject,
            "body": text_body,
            "htmlBody": html_body,
            "attachments": [
                {
                    "content": audio_base64,
                    "filename": "hn-tech-podcast.mp3",
                    "mimeType": "audio/mpeg",
                }
            ],
        },
    )
    message_id = getattr(result, "id", None)
    return str(message_id) if message_id is not None else ""


@workflows.workflow.define(name="hn-tech-digest", on_behalf_of=True)
@uses_connectors(gmail)
class HnTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, data: DigestInput) -> DigestOutput:
        stories = await fetch_hn_front_page(data.max_stories)
        tech_stories = await select_tech_stories(stories)
        digest = await build_digest(tech_stories)

        if data.dry_run:
            return DigestOutput(
                subject=digest.subject,
                text_body=digest.text_body,
                html_body=digest.html_body,
                podcast_script=digest.podcast_script,
                selected_stories=tech_stories,
                sent=False,
            )

        audio_base64 = await generate_podcast_audio(digest.podcast_script)
        audio_bytes = len(base64.b64decode(audio_base64))
        message_id = await send_digest_email(
            to=data.to,
            subject=digest.subject,
            text_body=digest.text_body,
            html_body=digest.html_body,
            audio_base64=audio_base64,
        )
        return DigestOutput(
            subject=digest.subject,
            text_body=digest.text_body,
            html_body=digest.html_body,
            podcast_script=digest.podcast_script,
            selected_stories=tech_stories,
            audio_bytes=audio_bytes,
            sent=True,
            message_id=message_id or None,
        )
