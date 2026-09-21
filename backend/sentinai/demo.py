"""Synthetic demo corpus generator.

Produces a realistic-looking but entirely fictional stream of public posts so that every
dashboard view (heatmap, trends + spikes + events, network graph, explainability, severity,
bot breakdown, review queue) has data without an X API key.  No real users or real posts
are included; usernames and IDs are generated.

The generator injects structure on purpose:
  * ~70 % benign posts across en/es/fr/pt/de/hi/ar (including AAVE / Hinglish benign text)
  * hostile posts with a realistic severity mix, obfuscated variants and emoji
  * counter-speech quote-tweets and replies (Module 6)
  * reply / retweet / quote cascades (Module 9), with a few coordinated bot clusters (Module 10)
  * two "event" spikes aligned with synthetic news events (Module 8)
  * a handful of multimodal posts (alt-text memes) (Module 5)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from sqlalchemy import func, select

from sentinai.config import get_settings
from sentinai.schemas import Author, Engagement, MediaAttachment, Post, ReferencedPost
from sentinai.storage.db import init_db, session_scope
from sentinai.storage.models import PostRow

# --------------------------------------------------------------------------------------------
# text banks (fictional; hostile examples are the kind of content the platform detects)
# --------------------------------------------------------------------------------------------

BENIGN = {
    "en": [
        "Had a lovely weekend hiking with my family, the weather was perfect!",
        "Just finished my thesis draft on urban transit policy — feeling relieved.",
        "Anyone know a good plumber near downtown? Kitchen sink is leaking again.",
        "Congrats to the whole team on shipping the release today 🎉",
        "The lecture on medieval trade routes was surprisingly fascinating.",
        "Our neighborhood cleanup is Saturday morning, bring gloves and friends!",
        "Reading a great novel about immigrants building a life in 1920s Chicago.",
        "The Ramadan iftar at the community center was packed, amazing food from everyone.",
        "Diwali lights on our street look incredible this year 🪔",
        "Shabbat dinner with the neighbors tonight, first time hosting!",
        "Proud of my daughter for finishing her first 5k this morning.",
        "we finna be at the cookout all day, y'all pull up fr",
        "bruh this new album is lit, deadass been on repeat all week",
        "my mama been cooking since 6am, ain't nobody leaving hungry today",
        "Interesting study on how remote work changed commuting patterns in 2024.",
        "The new Nigerian restaurant on 5th has the best jollof in the city, no debate.",
        "Learning Arabic calligraphy this semester and it is humbling.",
        "Mexican independence day parade downtown was beautiful, so much music.",
        "Local mosque and church ran a joint food bank drive — 4 tons collected.",
        "Coffee, rain, and a stack of grading. Sunday mood.",
        "Big thanks to the Sikh volunteers who fed hundreds after the storm. 🙏",
        "Our Chinese New Year potluck had 30 dishes. I regret nothing.",
        "Research shows hate speech against Asian people rose sharply in 2020 — new paper out today.",
        "The documentary on Syrian refugee doctors rebuilding clinics is worth your time.",
        "yaar the traffic in Bangalore today was ekdum crazy, 2 hours for 8 km",
        "bhai kal ka match dekha? bilkul mast tha",
        "yalla habibi, the shawarma at that place on 3rd is kteer good wallah",
        "cheers mate, proper knackered after that shift but the lads made it a laugh",
        "órale carnal, la neta the tacos at that troca on 5th are the best in town",
        "Great panel today on countering antisemitism and islamophobia together. Solidarity works.",
    ],
    "es": [
        "Qué buen partido anoche, el estadio estaba lleno de familias.",
        "Hoy empiezan las clases de cerámica en el centro comunitario, apúntense.",
        "La feria de comida latina del barrio estuvo increíble, gracias a todos.",
        "Terminé el informe a tiempo, ahora sí un café tranquilo.",
        "El documental sobre migrantes venezolanos en Bogotá es muy conmovedor.",
        "Feliz día de la independencia a todos los mexicanos 🇲🇽",
    ],
    "fr": [
        "Superbe expo au musée ce week-end, allez-y avant la fermeture.",
        "Le marché du dimanche avait des fraises incroyables.",
        "Merci aux bénévoles de la mosquée et de l'église pour la collecte alimentaire.",
        "Le débat sur la politique migratoire mérite mieux que des slogans.",
    ],
    "pt": [
        "O jogo de ontem foi incrível, que virada no segundo tempo!",
        "Festa junina da escola foi um sucesso, obrigado a todas as famílias.",
        "A nova padaria portuguesa do bairro tem o melhor pastel de nata.",
        "Documentário sobre imigrantes haitianos em São Paulo é muito bom.",
    ],
    "de": [
        "Das Konzert gestern im Park war wunderbar, danke an alle Helfer.",
        "Heute Nachbarschaftsfest mit Essen aus zwölf Ländern — großartig.",
        "Interessanter Vortrag über Migration und Arbeitsmarkt an der Uni.",
        "Die Moschee und die Kirche haben gemeinsam ein Winterfest organisiert.",
    ],
    "hi": [
        "आज मौसम बहुत अच्छा है, परिवार के साथ पार्क जा रहे हैं।",
        "सब धर्मों के लोग मिलकर बाढ़ पीड़ितों की मदद कर रहे हैं, गर्व है।",
        "दिवाली की शुभकामनाएं सभी को 🪔",
        "क्रिकेट मैच शानदार था, आखिरी ओवर तक सांस अटकी रही।",
    ],
    "ar": [
        "الطقس اليوم جميل، خرجنا مع العائلة إلى الحديقة.",
        "شكراً لمتطوعي المسجد والكنيسة على حملة جمع الطعام.",
        "المعرض الفني في وسط المدينة يستحق الزيارة هذا الأسبوع.",
        "مباراة الأمس كانت رائعة، الجمهور كان مذهلاً.",
    ],
}

# (text, lang) — hostile examples spanning the severity taxonomy.  Templates with {G} are
# filled with a group phrase from GROUPS so many demographics are covered.
HOSTILE_TEMPLATES = [
    # level 1 stereotyping
    ("{G} are all good at math, of course he got the job", "en", 1),
    ("typical {G}, what do you expect from {G}", "en", 1),
    ("{G} are naturally lazy, everyone knows it", "en", 1),
    ("{G} can't be trusted with anything, not surprised", "en", 1),
    ("todos los {GES} son iguales, no se puede confiar en ellos", "es", 1),
    ("tous les {GFR} sont pareils, jamais fiables", "fr", 1),
    ("alle {GDE} sind gleich, typisch", "de", 1),
    ("सारे {GHI} एक जैसे होते हैं", "hi", 1),
    # level 2 dehumanization / exclusion
    ("{G} don't belong here. Send them all back #StopTheInvasion", "en", 2),
    ("the {G} invasion is destroying our country, deport them all", "en", 2),
    ("{G} are like cockroaches, they breed like rats and infest our cities", "en", 2),
    ("{G} are not even human, they are a plague on this nation 🐀", "en", 2),
    ("no more {G}. ban them all and keep them out", "en", 2),
    ("los {GES} son una plaga, que se vayan de aquí, no pertenecen aquí", "es", 2),
    ("les {GFR} n'ont rien à faire ici, qu'ils rentrent chez eux", "fr", 2),
    ("os {GPT} são uns animais, não pertencem aqui", "pt", 2),
    ("die {GDE} gehören nicht hierher, alle abschieben", "de", 2),
    ("ये {GHI} घुसपैठिए हैं, इनको बाहर निकालो", "hi", 2),
    ("{GAR} حشرات، اطردوهم من بلدنا", "ar", 2),
    # level 3 slurs / harassment
    ("filthy {G} are a cancer on this country", "en", 3),
    ("I hate {G} so much, disgusting {G} scum everywhere", "en", 3),
    ("shut up you stupid {G} pig, nobody wants you here", "en", 3),
    ("f*ck the {G}, worthless {G} trash", "en", 3),
    ("malditos {GES} de mierda, odio a los {GES}", "es", 3),
    ("sales {GFR} de merde, je déteste les {GFR}", "fr", 3),
    ("odeio os {GPT}, {GPT} nojentos", "pt", 3),
    ("ich hasse die {GDE}, dreckiges {GDE} pack", "de", 3),
    ("साले {GHI} कुत्ते, मुझे {GHI} से नफरत है", "hi", 3),
    ("أكره كل {GAR}، {GAR} كلاب", "ar", 3),
    # level 4 incitement
    ("we should attack every {G} neighborhood, time to take up arms", "en", 4),
    ("kill all the {G}, every last one of them", "en", 4),
    ("death to {G}. burn their temples down", "en", 4),
    ("{G} deserve to die, someone needs to shoot them all", "en", 4),
    ("k1ll all the {G} 🔫🔫", "en", 4),
    ("hay que matar a todos los {GES}, muerte a los {GES}", "es", 4),
    ("il faut tuer tous les {GFR}, mort aux {GFR}", "fr", 4),
    ("matem todos os {GPT}, morte aos {GPT}", "pt", 4),
    ("tötet alle {GDE}, tod den {GDE}", "de", 4),
    ("{GHI} को मार दो, सबको जला दो", "hi", 4),
    ("اقتلوا كل {GAR}، الموت ل{GAR}", "ar", 4),
]

GROUPS = [
    # en, es, fr, pt, de, hi, ar  (neutral group nouns)
    ("muslims", "musulmanes", "musulmans", "muçulmanos", "muslime", "मुसलमान", "المسلمين"),
    ("jews", "judíos", "juifs", "judeus", "juden", "यहूदी", "اليهود"),
    ("christians", "cristianos", "chrétiens", "cristãos", "christen", "ईसाई", "المسيحيين"),
    ("hindus", "hindúes", "hindous", "hindus", "hindus", "हिंदू", "الهندوس"),
    ("sikhs", "sijs", "sikhs", "sikhs", "sikhs", "सिख", "السيخ"),
    ("black people", "negros", "noirs", "negros", "schwarze", "अश्वेत", "السود"),
    ("asians", "asiáticos", "asiatiques", "asiáticos", "asiaten", "एशियाई", "الآسيويين"),
    ("arabs", "árabes", "arabes", "árabes", "araber", "अरब", "العرب"),
    ("latinos", "latinos", "latinos", "latinos", "latinos", "लातीनी", "اللاتينيين"),
    ("white people", "blancos", "blancs", "brancos", "weiße", "गोरे", "البيض"),
    ("mexicans", "mexicanos", "mexicains", "mexicanos", "mexikaner", "मेक्सिकन", "المكسيكيين"),
    ("chinese people", "chinos", "chinois", "chineses", "chinesen", "चीनी", "الصينيين"),
    ("indians", "indios", "indiens", "indianos", "inder", "भारतीय", "الهنود"),
    ("pakistanis", "paquistaníes", "pakistanais", "paquistaneses", "pakistaner", "पाकिस्तानी", "الباكستانيين"),
    ("nigerians", "nigerianos", "nigérians", "nigerianos", "nigerianer", "नाइजीरियाई", "النيجيريين"),
    ("syrian refugees", "sirios", "syriens", "sírios", "syrer", "सीरियाई", "السوريين"),
    ("haitians", "haitianos", "haïtiens", "haitianos", "haitianer", "हैतीवासी", "الهايتيين"),
    ("immigrants", "inmigrantes", "immigrés", "imigrantes", "ausländer", "शरणार्थी", "المهاجرين"),
    ("palestinians", "palestinos", "palestiniens", "palestinos", "palästinenser", "फिलिस्तीनी", "الفلسطينيين"),
    ("israelis", "israelíes", "israéliens", "israelenses", "israelis", "इजरायली", "الإسرائيليين"),
    ("ukrainians", "ucranianos", "ukrainiens", "ucranianos", "ukrainer", "यूक्रेनी", "الأوكرانيين"),
    ("russians", "rusos", "russes", "russos", "russen", "रूसी", "الروس"),
    ("somalis", "somalíes", "somaliens", "somalis", "somalier", "सोमाली", "الصوماليين"),
    ("turks", "turcos", "turcs", "turcos", "türken", "तुर्क", "الأتراك"),
]

COUNTERSPEECH = [
    "This is blatantly racist. How is this account still up? Reported.",
    "Imagine posting this in 2025. The bigotry against {G} in my mentions is exhausting.",
    "Quoting so people can see the kind of hate {G} deal with every single day. Not okay.",
    "As someone from that community: this is exactly the dehumanizing rhetoric we warned about.",
    "Documenting this for our research on hate speech targeting {G}. Screenshot taken.",
    "Stop the hate. There is no place for this kind of talk about {G}.",
    "Esto es racismo puro. Denunciado.",
    "C'est du racisme pur et simple, honte à vous.",
    "Das ist blanker Rassismus, gemeldet.",
    "ये सरासर गलत है, नफरत मत फैलाओ। रिपोर्ट किया।",
    "هذا كلام عنصري مرفوض، تم الإبلاغ.",
]

SUPPORT_REPLIES = ["exactly", "facts", "this 💯", "so true", "finally someone said it", "based", "say it louder", "100% agree"]
QUERY_REPLIES = ["source?", "why do you think that?", "what do you mean by this", "is this real?"]
OFFENSIVE = ["shut up you absolute idiot", "what a pathetic clown take", "you're a moron and everyone knows it", "stfu loser", "eres un idiota", "t'es qu'un abruti", "du bist so ein Idiot", "बकवास बंद करो बेवकूफ"]

MEME_ALT = [
    "meme with text: THEY ARE REPLACING US. wake up",
    "cartoon of {G} depicted as rats swarming a city, caption: INFESTATION",
    "image with text 1488 and a black sun symbol",
    "photo of a peaceful protest with sign: refugees welcome",
    "screenshot of a news article about the election results",
]

FIRST = ["alex", "sam", "jordan", "maria", "li", "omar", "priya", "ana", "chris", "fatima", "ivan", "yuki", "nadia", "tom", "lena", "raj", "sara", "luis", "emma", "ahmed", "kofi", "mei", "diego", "hana"]
LAST = ["writes", "daily", "real", "official", "online", "thoughts", "hq", "here", "now", "world", "media", "voice", "news", "talks", "life", "watch"]

EVENTS = [
    (-62, "Contested general election result announced", "election", ["nationality:immigrants_generic", "ethnicity:hispanic_latino", "nationality:mexican"]),
    (-31, "Terror attack in European capital; suspect's background reported", "conflict", ["religion:islam", "ethnicity:middle_eastern", "nationality:syrian"]),
    (-14, "International football final marred by crowd violence", "sport", ["ethnicity:black", "nationality:french"]),
    (-5, "Border policy debate in parliament", "politics", ["nationality:immigrants_generic"]),
]


# --------------------------------------------------------------------------------------------


def _fill(template: str, group: tuple[str, ...]) -> str:
    en, es, fr, pt, de, hi, ar = group
    return template.replace("{GES}", es).replace("{GFR}", fr).replace("{GPT}", pt).replace("{GDE}", de).replace("{GHI}", hi).replace("{GAR}", ar).replace("{G}", en)


def _author(rng: random.Random, i: int, now: datetime, bot: bool = False) -> Author:
    if bot:
        uname = f"{rng.choice(FIRST)}{rng.randint(10000, 99999)}"
        return Author(id=f"u{i:05d}", username=uname, display_name=None, created_at=now - timedelta(days=rng.randint(3, 60)), followers_count=rng.randint(0, 40), following_count=rng.randint(800, 4000), tweet_count=rng.randint(5000, 40000), description=None, location=None, has_default_profile_image=rng.random() < 0.8)
    uname = f"{rng.choice(FIRST)}_{rng.choice(LAST)}{rng.randint(1, 99) if rng.random() < 0.5 else ''}"
    return Author(id=f"u{i:05d}", username=uname, display_name=uname.replace("_", " ").title(), created_at=now - timedelta(days=rng.randint(200, 4000)), followers_count=int(rng.lognormvariate(5, 1.4)), following_count=int(rng.lognormvariate(5, 1.0)), tweet_count=int(rng.lognormvariate(6.5, 1.2)), description=rng.choice(["coffee, books, city life", "researcher · views my own", "dad of two · runner", "🇲🇽🇺🇸 · music · food", "journalist covering local politics", "just here for the memes"]), location=rng.choice(["Chicago", "London", "Berlin", "Madrid", "Mumbai", "Cairo", "São Paulo", "Toronto", "Paris", "Lagos", None]), has_default_profile_image=rng.random() < 0.05)


def generate(n_posts: int = 1600, seed: int = 42, days: int = 90, now: datetime | None = None) -> tuple[list[Post], list[dict]]:
    rng = random.Random(seed)
    now = now or datetime.utcnow()
    n_authors = max(60, n_posts // 8)
    n_bots = max(6, n_authors // 10)
    authors = [_author(rng, i, now, bot=(i >= n_authors - n_bots)) for i in range(n_authors)]
    humans, bots = authors[: n_authors - n_bots], authors[n_authors - n_bots :]
    # a few "hostile" human accounts that produce most of the hate, and amplifiers
    hostile_humans = rng.sample(humans, max(6, len(humans) // 8))
    # a handful of prolific "super-posters" produce most of the hostile volume (heavy-tailed, as on real
    # platforms) — this also guarantees accounts above the 50-post minimum for reliable account scores
    hostile_weights = [14 if i < 4 else 1 for i in range(len(hostile_humans))]
    amplifiers = rng.sample([h for h in humans if h not in hostile_humans], max(8, len(humans) // 6))
    counter_speakers = rng.sample([h for h in humans if h not in hostile_humans and h not in amplifiers], max(8, len(humans) // 6))

    # daily intensity with two event-driven spikes
    day_weights = []
    for d in range(days):
        offset = -(days - 1 - d)
        w = 1.0 + 0.15 * rng.random()
        for ev_off, _, _, _ in EVENTS[:3]:
            if 0 <= offset - ev_off <= 4:
                w += 2.2 * (0.55 ** (offset - ev_off))
        day_weights.append(w)

    posts: list[Post] = []
    pid = 1000000

    def next_id() -> str:
        nonlocal pid
        pid += 1
        return str(pid)

    def ts_for_day(d: int) -> datetime:
        base = now - timedelta(days=(days - 1 - d))
        return base.replace(hour=rng.randint(0, 23), minute=rng.randint(0, 59), second=rng.randint(0, 59), microsecond=0)

    langs = ["en"] * 12 + ["es"] * 3 + ["fr"] * 2 + ["pt"] * 2 + ["de"] * 2 + ["hi"] * 2 + ["ar"] * 2
    for _ in range(n_posts):
        d = rng.choices(range(days), weights=day_weights, k=1)[0]
        ts = ts_for_day(d)
        roll = rng.random()
        # spikes skew toward hostile content
        hostile_p = 0.22 + 0.25 * (day_weights[d] - 1.0) / 3.0
        if roll < hostile_p:
            tmpl, lang, level = rng.choice(HOSTILE_TEMPLATES)
            group = rng.choice(GROUPS)
            # event-related targeting
            for ev_off, _, _, targets in EVENTS[:3]:
                if 0 <= -(days - 1 - d) - ev_off <= 4 and rng.random() < 0.6:
                    if "religion:islam" in targets:
                        group = GROUPS[0] if rng.random() < 0.6 else GROUPS[15]
                    elif "nationality:mexican" in targets:
                        group = GROUPS[10] if rng.random() < 0.5 else GROUPS[17]
                    elif "ethnicity:black" in targets:
                        group = GROUPS[5]
            text = _fill(tmpl, group)
            author = rng.choices(hostile_humans, weights=hostile_weights, k=1)[0] if rng.random() < 0.75 else rng.choice(bots)
            media = []
            if rng.random() < 0.06:
                media = [MediaAttachment(media_key=f"m{pid}", type="photo", alt_text=_fill(rng.choice(MEME_ALT[:3]), group))]
            root = Post(id=next_id(), text=text, created_at=ts, author_id=author.id, author=author, lang=lang, engagement=Engagement(like_count=int(rng.lognormvariate(2.5, 1.2)), retweet_count=int(rng.lognormvariate(1.5, 1.2)), reply_count=rng.randint(0, 30), quote_count=rng.randint(0, 10)), media=media)
            posts.append(root)
            # cascade: amplification (retweets / support replies), counter-speech quotes, bot retweets
            n_children = rng.choices([0, 1, 2, 3, 5, 8], weights=[35, 25, 15, 12, 8, 5], k=1)[0] + (2 if level >= 3 else 0)
            for _c in range(n_children):
                cts = ts + timedelta(minutes=rng.randint(2, 600))
                kind = rng.random()
                if kind < 0.35:
                    a = rng.choice(bots) if rng.random() < 0.55 else rng.choice(amplifiers)
                    posts.append(Post(id=next_id(), text=f"RT @{author.username}: {text}", created_at=cts, author_id=a.id, author=a, lang=lang, referenced=[ReferencedPost(type="retweeted", id=root.id)], conversation_id=root.id))
                elif kind < 0.55:
                    a = rng.choice(amplifiers)
                    posts.append(Post(id=next_id(), text=rng.choice(SUPPORT_REPLIES), created_at=cts, author_id=a.id, author=a, lang="en", referenced=[ReferencedPost(type="replied_to", id=root.id)], conversation_id=root.id, in_reply_to_user_id=author.id))
                elif kind < 0.85:
                    a = rng.choice(counter_speakers)
                    posts.append(Post(id=next_id(), text=_fill(rng.choice(COUNTERSPEECH), group), created_at=cts, author_id=a.id, author=a, lang="en", referenced=[ReferencedPost(type="quoted" if rng.random() < 0.6 else "replied_to", id=root.id)], conversation_id=root.id, engagement=Engagement(like_count=int(rng.lognormvariate(3, 1.0)), retweet_count=int(rng.lognormvariate(2, 1.0)))))
                else:
                    a = rng.choice(humans)
                    posts.append(Post(id=next_id(), text=rng.choice(QUERY_REPLIES), created_at=cts, author_id=a.id, author=a, lang="en", referenced=[ReferencedPost(type="replied_to", id=root.id)], conversation_id=root.id))
        elif roll < hostile_p + 0.05:
            author = rng.choice(humans)
            posts.append(Post(id=next_id(), text=rng.choice(OFFENSIVE), created_at=ts, author_id=author.id, author=author, lang="en"))
        else:
            lang = rng.choice(langs)
            author = rng.choice(humans) if rng.random() < 0.9 else rng.choice(bots)
            text = rng.choice(BENIGN[lang])
            media = []
            if rng.random() < 0.03:
                media = [MediaAttachment(media_key=f"m{pid}", type="photo", alt_text=rng.choice(MEME_ALT[3:]))]
            posts.append(Post(id=next_id(), text=text, created_at=ts, author_id=author.id, author=author, lang=lang, engagement=Engagement(like_count=int(rng.lognormvariate(2.0, 1.3)), retweet_count=int(rng.lognormvariate(0.8, 1.2)), reply_count=rng.randint(0, 12)), media=media))

    # coordinated bot cluster: near-duplicate hostile posts at regular intervals
    cluster_text = _fill(HOSTILE_TEMPLATES[9][0], GROUPS[17])
    for i, b in enumerate(bots[: max(3, len(bots) // 2)]):
        for k in range(6):
            ts = now - timedelta(days=rng.randint(0, 20), hours=k * 4, minutes=i)
            variant = cluster_text + (" !!!" if k % 2 else "") + (f" #{rng.choice(['SaveOurCountry', 'WakeUp', 'Borders'])}" if k % 3 == 0 else "")
            posts.append(Post(id=next_id(), text=variant, created_at=ts, author_id=b.id, author=b, lang="en"))

    events = [{"date": (now + timedelta(days=off)).strftime("%Y-%m-%d"), "title": title, "category": cat, "source": "demo", "related_targets": targets} for off, title, cat, targets in EVENTS]
    posts.sort(key=lambda p: p.created_at)
    return posts, events


def seed(n_posts: int | None = None, seed_value: int | None = None) -> int:
    from sentinai.analytics.accounts import score_all_accounts
    from sentinai.analytics.bots import score_all_authors
    from sentinai.analytics.trends import store_events
    from sentinai.hitl import enqueue_candidates
    from sentinai.ingestion.worker import IngestionWorker

    settings = get_settings()
    init_db()
    posts, events = generate(n_posts or 1600, seed_value if seed_value is not None else settings.demo_seed)
    with session_scope() as session:
        worker = IngestionWorker(session)
        n = 0
        for i in range(0, len(posts), 200):
            n += worker.process(posts[i : i + 200])
        store_events(session, events)
        score_all_authors(session)
        score_all_accounts(session)
        enqueue_candidates(session, limit=100)
    return n


def seed_if_empty() -> int:
    init_db()
    with session_scope() as session:
        if (session.scalar(select(func.count()).select_from(PostRow)) or 0) > 0:
            return 0
    return seed()
