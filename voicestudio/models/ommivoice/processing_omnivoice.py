"""Processor class for OmniVoice."""

import bisect
import difflib
import random
import re
import unicodedata
from functools import lru_cache

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)


DEFAULT_AUDIO_TOKENIZER_ID = "eustlb/higgs-audio-v2-tokenizer"

# Lowercase language name -> ISO 639-3 (or ISO 639-1, where one exists) code accepted between the
# `<|lang_start|>` / `<|lang_end|>` markers.
LANG_NAME_TO_ID = {
    "abadi": "kbt",
    "abkhazian": "ab",
    "abron": "abr",
    "abua": "abn",
    "adamawa fulfulde": "fub",
    "adyghe": "ady",
    "afade": "aal",
    "afrikaans": "af",
    "agwagwune": "yay",
    "aja (benin)": "ajg",
    "akebu": "keu",
    "alago": "ala",
    "albanian": "sq",
    "algerian arabic": "arq",
    "algerian saharan arabic": "aao",
    "ambo-pasco quechua": "qva",
    "ambonese malay": "abs",
    "amdo tibetan": "adx",
    "amharic": "am",
    "anaang": "anw",
    "angika": "anp",
    "antankarana malagasy": "xmv",
    "aragonese": "an",
    "arbëreshë albanian": "aae",
    "arequipa-la unión quechua": "qxu",
    "armenian": "hy",
    "ashe": "ahs",
    "ashéninka perené": "prq",
    "askopan": "eiv",
    "assamese": "as",
    "asturian": "ast",
    "atayal": "tay",
    "awak": "awo",
    "ayacucho quechua": "quy",
    "azerbaijani": "az",
    "baatonum": "bba",
    "bacama": "bcy",
    "bade": "bde",
    "bafia": "ksf",
    "bafut": "bfd",
    "bagirmi fulfulde": "fui",
    "bago-kusuntu": "bqg",
    "baharna arabic": "abv",
    "bakoko": "bkh",
    "balanta-ganja": "bjt",
    "balti": "bft",
    "bamenyam": "bce",
    "bamun": "bax",
    "bangwinji": "bsj",
    "banjar": "bjn",
    "bankon": "abb",
    "baoulé": "bci",
    "bara malagasy": "bhr",
    "barok": "bjk",
    "basa (cameroon)": "bas",
    "basa (nigeria)": "bzw",
    "bashkir": "ba",
    "basque": "eu",
    "batak mandailing": "btm",
    "batanga": "bnm",
    "bateri": "btv",
    "bats": "bbl",
    "bayot": "bda",
    "bebele": "beb",
    "belarusian": "be",
    "bengali": "bn",
    "betawi": "bew",
    "bhili": "bhb",
    "bhojpuri": "bho",
    "bilur": "bxf",
    "bima": "bhp",
    "bodo": "brx",
    "boghom": "bux",
    "bokyi": "bky",
    "bomu": "bmq",
    "bondei": "bou",
    "borgu fulfulde": "fue",
    "bosnian": "bs",
    "brahui": "brh",
    "braj": "bra",
    "breton": "br",
    "buduma": "bdm",
    "buginese": "bug",
    "bukharic": "bhh",
    "bulgarian": "bg",
    "bulu (cameroon)": "bum",
    "bundeli": "bns",
    "bunun": "bnn",
    "bura-pabir": "bwr",
    "burak": "bys",
    "burmese": "my",
    "burushaski": "bsk",
    "cacaloxtepec mixtec": "miu",
    "cajatambo north lima quechua": "qvl",
    "cakfem-mushere": "cky",
    "cameroon pidgin": "wes",
    "campidanese sardinian": "sro",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "cen": "cen",
    "central kurdish": "ckb",
    "central nahuatl": "nhn",
    "central pame": "pbs",
    "central pashto": "pst",
    "central puebla nahuatl": "ncx",
    "central tarahumara": "tar",
    "central yupik": "esu",
    "central-eastern niger fulfulde": "fuq",
    "chadian arabic": "shu",
    "chichewa": "ny",
    "chichicapan zapotec": "zpv",
    "chiga": "cgg",
    "chimalapa zoque": "zoh",
    "chimborazo highland quichua": "qug",
    "chinese": "zh",
    "chiquián ancash quechua": "qxa",
    "chitwania tharu": "the",
    "chokwe": "cjk",
    "chuvash": "cv",
    "cibak": "ckl",
    "coastal konjo": "kjc",
    "copainalá zoque": "zoc",
    "cornish": "kw",
    "corongo ancash quechua": "qwa",
    "croatian": "hr",
    "cross river mbembe": "mfn",
    "cuyamecalco mixtec": "xtu",
    "czech": "cs",
    "dadiya": "dbd",
    "dagbani": "dag",
    "dameli": "dml",
    "danish": "da",
    "dargwa": "dar",
    "dazaga": "dzg",
    "deccan": "dcc",
    "degema": "deg",
    "dera (nigeria)": "kna",
    "dghwede": "dgh",
    "dhatki": "mki",
    "dhivehi": "dv",
    "dhofari arabic": "adf",
    "dijim-bwilim": "cfa",
    "dogri": "dgo",
    "domaaki": "dmk",
    "dotyali": "dty",
    "duala": "dua",
    "dutch": "nl",
    "dũya": "ldb",
    "dyula": "dyu",
    "eastern balochi": "bgp",
    "eastern bolivian guaraní": "gui",
    "eastern egyptian bedawi arabic": "avl",
    "eastern krahn": "kqo",
    "eastern mari": "mhr",
    "eastern yiddish": "ydd",
    "ebrié": "ebr",
    "eggon": "ego",
    "egyptian arabic": "arz",
    "ejagham": "etu",
    "eleme": "elm",
    "eloyi": "afo",
    "embu": "ebu",
    "english": "en",
    "erzya": "myv",
    "esan": "ish",
    "esperanto": "eo",
    "estonian": "et",
    "eton (cameroon)": "eto",
    "ewondo": "ewo",
    "extremaduran": "ext",
    "fang (equatorial guinea)": "fan",
    "fanti": "fat",
    "farefare": "gur",
    "fe'fe'": "fmp",
    "filipino": "fil",
    "filomena mata-coahuitlán totonac": "tlp",
    "finnish": "fi",
    "fipa": "fip",
    "french": "fr",
    "fulah": "ff",
    "galician": "gl",
    "gambian wolof": "wof",
    "ganda": "lg",
    "garhwali": "gbm",
    "gawar-bati": "gwt",
    "gawri": "gwc",
    "gbagyi": "gbr",
    "gbari": "gby",
    "geji": "gyz",
    "gen": "gej",
    "georgian": "ka",
    "german": "de",
    "geser-gorom": "ges",
    "gheg albanian": "aln",
    "ghomálá'": "bbj",
    "gidar": "gid",
    "glavda": "glw",
    "goan konkani": "gom",
    "goaria": "gig",
    "goemai": "ank",
    "gola": "gol",
    "greek": "el",
    "guarani": "gn",
    "guduf-gava": "gdf",
    "guerrero amuzgo": "amu",
    "gujarati": "gu",
    "gujari": "gju",
    "gulf arabic": "afb",
    "gurgula": "ggg",
    "gusii": "guz",
    "gusilay": "gsl",
    "gweno": "gwe",
    "güilá zapotec": "ztu",
    "hadothi": "hoj",
    "hahon": "hah",
    "haitian": "ht",
    "hakha chin": "cnh",
    "hakö": "hao",
    "halia": "hla",
    "hausa": "ha",
    "hawaiian": "haw",
    "hazaragi": "haz",
    "hebrew": "he",
    "hemba": "hem",
    "herero": "hz",
    "highland konjo": "kjk",
    "hijazi arabic": "acw",
    "hindi": "hi",
    "huarijio": "var",
    "huautla mazatec": "mau",
    "huaxcaleca nahuatl": "nhq",
    "huba": "hbb",
    "huitepec mixtec": "mxs",
    "hula": "hul",
    "hungarian": "hu",
    "hunjara-kaina ke": "hkk",
    "hwana": "hwo",
    "ibibio": "ibb",
    "icelandic": "is",
    "idakho-isukha-tiriki": "ida",
    "idoma": "idu",
    "igbo": "ig",
    "igo": "ahl",
    "ikposo": "kpo",
    "ikwere": "ikw",
    "imbabura highland quichua": "qvi",
    "indonesian": "id",
    "indus kohistani": "mvy",
    "interlingua (international auxiliary language association)": "ia",
    "inupiaq": "ik",
    "irish": "ga",
    "iron ossetic": "os",
    "isekiri": "its",
    "isoko": "iso",
    "italian": "it",
    "ito": "itw",
    "itzá": "itz",
    "ixtayutla mixtec": "vmj",
    "izon": "ijc",
    "jambi malay": "jax",
    "japanese": "ja",
    "jaqaru": "jqr",
    "jauja wanca quechua": "qxw",
    "jaunsari": "jns",
    "javanese": "jv",
    "jiba": "juo",
    "jju": "kaj",
    "judeo-moroccan arabic": "aju",
    "juxtlahuaca mixtec": "vmc",
    "kabardian": "kbd",
    "kabras": "lkb",
    "kabuverdianu": "kea",
    "kabyle": "kab",
    "kachi koli": "gjk",
    "kairak": "ckr",
    "kalabari": "ijn",
    "kalasha": "kls",
    "kalenjin": "kln",
    "kalkoti": "xka",
    "kamba": "kam",
    "kamo": "kcq",
    "kanauji": "bjj",
    "kanembu": "kbl",
    "kannada": "kn",
    "karekare": "kai",
    "kashmiri": "ks",
    "kathoriya tharu": "tkt",
    "kati": "bsh",
    "kazakh": "kk",
    "keiyo": "eyo",
    "khams tibetan": "khg",
    "khana": "ogo",
    "khetrani": "xhe",
    "khmer": "km",
    "khowar": "khw",
    "kinga": "zga",
    "kinnauri": "kfk",
    "kinyarwanda": "rw",
    "kirghiz": "ky",
    "kirya-konzəl": "fkk",
    "kochila tharu": "thq",
    "kohistani shina": "plk",
    "kohumono": "bcs",
    "kok borok": "trp",
    "kol (papua new guinea)": "kol",
    "kom (cameroon)": "bkm",
    "koma": "kmy",
    "konkani": "knn",
    "konzo": "koo",
    "korean": "ko",
    "korwa": "kfp",
    "kota (india)": "kfe",
    "koti": "eko",
    "kuanua": "ksd",
    "kuanyama": "kj",
    "kui (india)": "uki",
    "kulung (nigeria)": "bbu",
    "kuot": "kto",
    "kushi": "kuh",
    "kwambi": "kwm",
    "kwasio": "nmg",
    "lala-roba": "lla",
    "lamang": "hia",
    "lao": "lo",
    "larike-wakasihu": "alo",
    "lasi": "lss",
    "latgalian": "ltg",
    "latvian": "lv",
    "levantine arabic": "apc",
    "liana-seti": "ste",
    "liberia kpelle": "xpe",
    "liberian english": "lir",
    "libyan arabic": "ayl",
    "ligurian": "lij",
    "lijili": "mgi",
    "lingala": "ln",
    "lithuanian": "lt",
    "loarki": "lrk",
    "logooli": "rag",
    "logudorese sardinian": "src",
    "loja highland quichua": "qvj",
    "loloda": "loa",
    "longuda": "lnu",
    "loxicha zapotec": "ztp",
    "luba-lulua": "lua",
    "luo": "luo",
    "lushai": "lus",
    "luxembourgish": "lb",
    "maasina fulfulde": "ffm",
    "maba (chad)": "mde",
    "macedo-romanian": "rup",
    "macedonian": "mk",
    "mada (cameroon)": "mxu",
    "mafa": "maf",
    "maithili": "mai",
    "malay": "ms",
    "malayalam": "ml",
    "mali": "gcc",
    "malinaltepec me'phaa": "tcf",
    "maltese": "mt",
    "mandara": "tbf",
    "mandjak": "mfv",
    "manggarai": "mqy",
    "manipuri": "mni",
    "mansoanka": "msw",
    "manx": "gv",
    "maori": "mi",
    "marathi": "mr",
    "marghi central": "mrt",
    "marghi south": "mfm",
    "maria (india)": "mrr",
    "marwari (pakistan)": "mve",
    "masana": "mcn",
    "masikoro malagasy": "msh",
    "matsés": "mcf",
    "mazaltepec zapotec": "zpy",
    "mazatlán mazatec": "vmz",
    "mazatlán mixe": "mzl",
    "mbe": "mfo",
    "mbo (cameroon)": "mbo",
    "mbum": "mdd",
    "medumba": "byv",
    "mekeo": "mek",
    "meru": "mer",
    "mesopotamian arabic": "acm",
    "mewari": "mtr",
    "min nan chinese": "nan",
    "mingrelian": "xmf",
    "mitlatongo mixtec": "vmm",
    "miya": "mkf",
    "mokpwe": "bri",
    "moksha": "mdf",
    "mom jango": "ver",
    "mongolian": "mn",
    "moroccan arabic": "ary",
    "motu": "meu",
    "mpiemo": "mcx",
    "mpumpong": "mgg",
    "mundang": "mua",
    "mungaka": "mhk",
    "musey": "mse",
    "musgu": "mug",
    "musi": "mui",
    "naba": "mne",
    "najdi arabic": "ars",
    "nalik": "nal",
    "nawdm": "nmz",
    "ndonga": "ng",
    "neapolitan": "nap",
    "nepali": "npi",
    "ngamo": "nbh",
    "ngas": "anc",
    "ngiemboon": "nnh",
    "ngizim": "ngi",
    "ngomba": "jgo",
    "ngombale": "nla",
    "nigerian fulfulde": "fuv",
    "nigerian pidgin": "pcm",
    "nimadi": "noe",
    "nobiin": "fia",
    "north mesopotamian arabic": "ayp",
    "north moluccan malay": "max",
    "northern betsimisaraka malagasy": "bmm",
    "northern hindko": "hno",
    "northern kurdish": "kmr",
    "northern pame": "pmq",
    "northern pashto": "pbu",
    "northern uzbek": "uzn",
    "northwest gbaya": "gya",
    "norwegian": "no",
    "norwegian bokmål": "nb",
    "norwegian nynorsk": "nn",
    "notsi": "ncf",
    "nyankpa": "yes",
    "nyungwe": "nyu",
    "nzanyi": "nja",
    "nüpode huitoto": "hux",
    "occitan": "oc",
    "od": "odk",
    "odia": "ory",
    "odual": "odu",
    "omani arabic": "acx",
    "orizaba nahuatl": "nlv",
    "orma": "orc",
    "ormuri": "oru",
    "oromo": "om",
    "pahari-potwari": "phr",
    "paiwan": "pwn",
    "panjabi": "pa",
    "papuan malay": "pmy",
    "parkari koli": "kvx",
    "pedi": "nso",
    "pero": "pip",
    "persian": "fa",
    "petats": "pex",
    "phalura": "phl",
    "piemontese": "pms",
    "piya-kwonci": "piy",
    "plateau malagasy": "plt",
    "polish": "pl",
    "poqomam": "poc",
    "portuguese": "pt",
    "pulaar": "fuc",
    "pular": "fuf",
    "puno quechua": "qxp",
    "pushto": "ps",
    "pökoot": "pko",
    "qaqet": "byx",
    "quiotepec chinantec": "chq",
    "rana tharu": "thr",
    "rangi": "lag",
    "rapoisi": "kyx",
    "ratahan": "rth",
    "rayón zoque": "zor",
    "romanian": "ro",
    "romansh": "rm",
    "rombo": "rof",
    "rotokas": "roo",
    "rukai": "dru",
    "russian": "ru",
    "sacapulteco": "quv",
    "saidi arabic": "aec",
    "sakalava malagasy": "skg",
    "sakizaya": "szy",
    "saleman": "sau",
    "samba daka": "ccg",
    "samba leko": "ndi",
    "san felipe otlaltepec popoloca": "pow",
    "san francisco del mar huave": "hue",
    "san juan atzingo popoloca": "poe",
    "san martín itunyoso triqui": "trq",
    "san miguel el grande mixtec": "mig",
    "sansi": "ssi",
    "sanskrit": "sa",
    "santa ana de tusi pasco quechua": "qxt",
    "santa catarina albarradas zapotec": "ztn",
    "santali": "sat",
    "santiago del estero quichua": "qus",
    "saposa": "sps",
    "saraiki": "skr",
    "sardinian": "sc",
    "saya": "say",
    "sediq": "trv",
    "serbian": "sr",
    "seri": "sei",
    "shina": "scl",
    "shona": "sn",
    "siar-lak": "sjr",
    "sibe": "nco",
    "sicilian": "scn",
    "sihuas ancash quechua": "qws",
    "sikkimese": "sip",
    "sinaugoro": "snc",
    "sindhi": "sd",
    "sindhi bhil": "sbn",
    "sinhala": "si",
    "sinicahua mixtec": "xti",
    "sipacapense": "qum",
    "siwai": "siw",
    "slovak": "sk",
    "slovenian": "sl",
    "solos": "sol",
    "somali": "so",
    "soninke": "snk",
    "south giziga": "giz",
    "south ucayali ashéninka": "cpy",
    "southeastern nochixtlán mixtec": "mxy",
    "southern betsimisaraka malagasy": "bzc",
    "southern pashto": "pbt",
    "southern pastaza quechua": "qup",
    "soyaltepec mazatec": "vmp",
    "spanish": "es",
    "standard arabic": "arb",
    "standard moroccan tamazight": "zgh",
    "sudanese arabic": "apd",
    "sulka": "sua",
    "svan": "sva",
    "swahili": "sw",
    "swedish": "sv",
    "tae'": "rob",
    "tahaggart tamahaq": "thv",
    "taita": "dav",
    "tajik": "tg",
    "tamil": "ta",
    "tandroy-mahafaly malagasy": "tdx",
    "tangale": "tan",
    "tanosy malagasy": "txy",
    "tarok": "yer",
    "tatar": "tt",
    "tedaga": "tuq",
    "telugu": "te",
    "tem": "kdh",
    "teop": "tio",
    "tepeuxila cuicatec": "cux",
    "tepinapa chinantec": "cte",
    "tera": "ttr",
    "terei": "buo",
    "termanu": "twu",
    "tesaka malagasy": "tkg",
    "tetelcingo nahuatl": "nhg",
    "teutila cuicatec": "cut",
    "thai": "th",
    "tibetan": "bo",
    "tidaá mixtec": "mtx",
    "tidore": "tvo",
    "tigak": "tgc",
    "tigre": "tig",
    "tigrinya": "ti",
    "tilquiapan zapotec": "zts",
    "tinputz": "tpz",
    "tlacoapa me'phaa": "tpl",
    "tlacoatzintepec chinantec": "ctl",
    "tlingit": "tli",
    "toki pona": "tok",
    "tomoip": "tqp",
    "tondano": "tdn",
    "tonsea": "txs",
    "tooro": "ttj",
    "torau": "ttu",
    "torwali": "trw",
    "tsimihety malagasy": "xmw",
    "tsotso": "lto",
    "tswana": "tn",
    "tugen": "tuy",
    "tuki": "bag",
    "tula": "tul",
    "tulu": "tcy",
    "tunen": "tvu",
    "tungag": "lcm",
    "tunisian arabic": "aeb",
    "tupuri": "tui",
    "turkana": "tuv",
    "turkish": "tr",
    "turkmen": "tk",
    "tututepec mixtec": "mtu",
    "twi": "tw",
    "ubaghara": "byc",
    "uighur": "ug",
    "ukrainian": "uk",
    "umbundu": "umb",
    "upper sorbian": "hsb",
    "urdu": "ur",
    "ushojo": "ush",
    "uzbek": "uz",
    "vai": "vai",
    "vietnamese": "vi",
    "votic": "vot",
    "võro": "vro",
    "waci gbe": "wci",
    "wadiyara koli": "kxp",
    "waja": "wja",
    "wakhi": "wbl",
    "wanga": "lwg",
    "wapan": "juk",
    "warji": "wji",
    "welsh": "cy",
    "wemale": "weo",
    "western frisian": "fy",
    "western highland purepecha": "pua",
    "western juxtlahuaca mixtec": "jmx",
    "western maninkakan": "mlq",
    "western mari": "mrj",
    "western niger fulfulde": "fuh",
    "western panjabi": "pnb",
    "wolof": "wo",
    "wuzlam": "udl",
    "xanaguía zapotec": "ztg",
    "xhosa": "xh",
    "yace": "ekr",
    "yakut": "sah",
    "yalahatan": "jal",
    "yanahuanca pasco quechua": "qur",
    "yangben": "yav",
    "yaqui": "yaq",
    "yauyos quechua": "qux",
    "yekhee": "ets",
    "yiddish": "yi",
    "yidgha": "ydg",
    "yoruba": "yo",
    "yutanduchi mixtec": "mab",
    "zacatlán-ahuacatlán-tepetzintla nahuatl": "nhi",
    "zarma": "dje",
    "zaza": "zza",
    "zulu": "zu",
    "ömie": "aom",
}

LANG_NAMES = frozenset(LANG_NAME_TO_ID.keys())
LANG_IDS = frozenset(LANG_NAME_TO_ID.values())

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

# Each entry is one mutually exclusive voice-design category: a dict maps the English item to its Chinese
# equivalent, a set holds items that exist in one language only.
_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
    {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    },
    {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    },
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
_INSTRUCT_MUTUALLY_EXCLUSIVE = []
for _category in _INSTRUCT_CATEGORIES:
    if isinstance(_category, dict):
        _INSTRUCT_EN_TO_ZH.update(_category)
        _INSTRUCT_ZH_TO_EN.update({value: key for key, value in _category.items()})
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_category) | set(_category.values()))
    else:
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_category))

_INSTRUCT_ALL_VALID = (
    set(_INSTRUCT_EN_TO_ZH)
    | set(_INSTRUCT_ZH_TO_EN)
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-2]
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-1]
)
_INSTRUCT_VALID_EN = frozenset(item for item in _INSTRUCT_ALL_VALID if not _ZH_RE.search(item))
_INSTRUCT_VALID_ZH = frozenset(item for item in _INSTRUCT_ALL_VALID if _ZH_RE.search(item))

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

_SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
_CLOSING_MARKS = set("\"'“”‘’）]》>」】")
_END_PUNCTUATION = set(";:,.!?…)]}\"'“”‘’；：，。！？、）】") | {"……"}
_ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.",
    "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.",
    "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.", "Apr.",
    "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.",
    "fig.", "def.",
}  # fmt: skip


def _resolve_language(language: str | None) -> str | None:
    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    if language.lower() in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[language.lower()]
    logger.warning(
        f"Language '{language}' is not recognized. Use a language id (e.g. 'en', 'zh', 'ja') or a full language "
        f"name (e.g. 'English', 'Chinese'); see `OmniVoiceProcessor.supported_language_ids` and "
        f"`OmniVoiceProcessor.supported_language_names`. Falling back to language-agnostic mode."
    )
    return None


def _resolve_instruct(instruct: str | None, use_zh: bool = False) -> str | None:
    """
    Validates and normalizes a voice-design instruct string to a single language.

    Args:
        instruct (`str`, *optional*):
            Comma separated speaker attributes, e.g. `"female, low pitch, british accent"` or `"女，四川话"`.
            Half-width and full-width commas are both accepted regardless of language.
        use_zh (`bool`, *optional*, defaults to `False`):
            Whether to normalize every item to Chinese. A Chinese dialect item forces `True` and an English
            accent item forces `False`, since neither has a counterpart in the other language.

    Returns:
        `str` or `None`: The normalized instruct string.

    Raises:
        ValueError: If an item is unsupported, if a Chinese dialect and an English accent are combined, or if two
            items of the same category are combined.
    """
    if instruct is None:
        return None
    instruct = instruct.strip()
    if not instruct:
        return None

    raw_items = [item for item in re.split(r"\s*[,，]\s*", instruct) if item]

    unknown = []
    normalized = []
    for raw in raw_items:
        item = raw.strip().lower()
        if item in _INSTRUCT_ALL_VALID:
            normalized.append(item)
        else:
            suggestions = difflib.get_close_matches(item, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, item, suggestions[0] if suggestions else None))

    if unknown:
        lines = [
            f"  '{raw}' -> '{item}' (unsupported{f'; did you mean {suggestion!r}?' if suggestion else ''})"
            for raw, item, suggestion in unknown
        ]
        raise ValueError(
            f"Unsupported instruct items found in {instruct}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nUse only English or only Chinese instructs."
        )

    has_dialect = any(item.endswith("话") for item in normalized)
    has_accent = any(" accent" in item for item in normalized)
    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix a Chinese dialect and an English accent in a single instruct. Dialects are for Chinese "
            "speech, accents for English speech."
        )
    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    if use_zh:
        normalized = [_INSTRUCT_EN_TO_ZH.get(item, item) for item in normalized]
    else:
        normalized = [_INSTRUCT_ZH_TO_EN.get(item, item) for item in normalized]

    conflicts = []
    for category in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [item for item in normalized if item in category]
        if len(hits) > 1:
            conflicts.append(" vs ".join(f"'{item}'" for item in hits))
    if conflicts:
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(conflicts)
            + ". Each category (gender, age, pitch, style, accent, dialect) allows at most one item."
        )

    separator = "，" if any(_ZH_RE.search(item) for item in normalized) else ", "
    return separator.join(normalized)


def _combine_text(text: str, reference_text: str | None = None) -> str:
    combined = f"{reference_text.strip()} {text.strip()}" if reference_text else text.strip()
    combined = re.sub(r"[\r\n]+", "", combined)
    combined = combined.replace("（", "(").replace("）", ")")
    combined = re.sub(r"[ \t]+", " ", combined)
    # Chinese is written without word spacing, so a space next to a Han character is a formatting artifact.
    combined = re.sub(r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])", "", combined)
    return combined


def _add_punctuation(text: str) -> str:
    text = text.strip()
    if not text or text[-1] in _END_PUNCTUATION:
        return text
    return text + ("。" if _ZH_RE.search(text) else ".")


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """
    Tokenizes `text` with every non-verbal tag held out and tokenized on its own, so that a tag maps to the same
    ids whatever the surrounding script is.

    Args:
        text (`str`):
            Text possibly containing non-verbal tags such as `[laughter]`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the OmniVoice checkpoint.

    Returns:
        `torch.Tensor` of shape `(1, sequence_length)`: The token ids.
    """
    parts = []
    last_end = 0
    for match in _NONVERBAL_PATTERN.finditer(text):
        if match.start() > last_end:
            ids = tokenizer(text[last_end : match.start()], add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(match.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = match.end()
    if last_end < len(text):
        ids = tokenizer(text[last_end:], add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer(text, return_tensors="pt").input_ids
    return torch.tensor([[token_id for part in parts for token_id in part]], dtype=torch.long)


def _chunk_text_punctuation(text: str, chunk_len: int, min_chunk_len: int | None = None) -> list[str]:
    sentences = []
    current_sentence = []
    for token in text:
        if not current_sentence and sentences and (token in _SPLIT_PUNCTUATION or token in _CLOSING_MARKS):
            sentences[-1].append(token)
            continue
        current_sentence.append(token)
        if token in _SPLIT_PUNCTUATION:
            is_abbreviation = False
            if token == ".":
                stripped = "".join(current_sentence).strip()
                if stripped and stripped.split()[-1] in _ABBREVIATIONS:
                    is_abbreviation = True
            if not is_abbreviation:
                sentences.append(current_sentence)
                current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if current_chunk:
                merged_chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        merged_chunks.append(current_chunk)

    if min_chunk_len is not None:
        first_chunk_short = bool(merged_chunks) and len(merged_chunks[0]) < min_chunk_len
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short:
                final_chunks[-1].extend(chunk)
            elif len(chunk) >= min_chunk_len or not final_chunks:
                final_chunks.append(chunk)
            else:
                final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    return [chunk for chunk in ("".join(chunk).strip() for chunk in final_chunks) if chunk]


def _fade_and_pad_audio(
    audio: np.ndarray, pad_duration: float, fade_duration: float, sampling_rate: int
) -> np.ndarray:
    if audio.shape[-1] == 0:
        return audio

    processed = audio.copy()
    fade_samples = min(int(fade_duration * sampling_rate), processed.shape[-1] // 2)
    if fade_samples > 0:
        processed[..., :fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
        processed[..., -fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

    pad_samples = int(pad_duration * sampling_rate)
    if pad_samples > 0:
        silence = np.zeros((*processed.shape[:-1], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)
    return processed


def _cross_fade_chunks(chunks: list[np.ndarray], sampling_rate: int, silence_duration: float = 0.3) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]

    fade_samples = int(silence_duration * sampling_rate) // 3
    merged = chunks[0].copy()
    for chunk in chunks[1:]:
        fade_out = min(fade_samples, merged.shape[-1])
        if fade_out > 0:
            merged[..., -fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        faded_in = chunk.copy()
        fade_in = min(fade_samples, faded_in.shape[-1])
        if fade_in > 0:
            faded_in[..., :fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        silence = np.zeros((*merged.shape[:-1], fade_samples), dtype=np.float32)
        merged = np.concatenate([merged, silence, faded_in], axis=-1)
    return merged


class OmniVoiceDurationEstimator:
    r"""
    Estimates how many audio frames a text needs, by scoring every character with the relative speaking time of
    its script and rescaling by the observed speaking rate of a reference pair.

    The weights are relative to one Latin letter (`1.0`, roughly 40-50 ms): a logographic character takes about
    three times as long, an abugida cluster about twice, a diacritic no time at all.
    """

    weights = {
        "cjk": 3.0,
        "hangul": 2.5,
        "kana": 2.2,
        "ethiopic": 3.0,
        "yi": 3.0,
        "indic": 1.8,
        "thai_lao": 1.5,
        "khmer_myanmar": 1.8,
        "arabic": 1.5,
        "hebrew": 1.5,
        "latin": 1.0,
        "cyrillic": 1.0,
        "greek": 1.0,
        "armenian": 1.0,
        "georgian": 1.0,
        "punctuation": 0.5,
        "space": 0.2,
        "digit": 3.5,
        "mark": 0.0,
        "default": 1.0,
    }

    # (last code point of the block, weight key), ordered for `bisect`.
    ranges = [
        (0x02AF, "latin"), (0x03FF, "greek"), (0x052F, "cyrillic"), (0x058F, "armenian"),
        (0x05FF, "hebrew"), (0x077F, "arabic"), (0x089F, "arabic"), (0x08FF, "arabic"),
        (0x097F, "indic"), (0x09FF, "indic"), (0x0A7F, "indic"), (0x0AFF, "indic"),
        (0x0B7F, "indic"), (0x0BFF, "indic"), (0x0C7F, "indic"), (0x0CFF, "indic"),
        (0x0D7F, "indic"), (0x0DFF, "indic"), (0x0EFF, "thai_lao"), (0x0FFF, "indic"),
        (0x109F, "khmer_myanmar"), (0x10FF, "georgian"), (0x11FF, "hangul"), (0x137F, "ethiopic"),
        (0x139F, "ethiopic"), (0x13FF, "default"), (0x167F, "default"), (0x169F, "default"),
        (0x16FF, "default"), (0x171F, "default"), (0x173F, "default"), (0x175F, "default"),
        (0x177F, "default"), (0x17FF, "khmer_myanmar"), (0x18AF, "default"), (0x18FF, "default"),
        (0x194F, "indic"), (0x19DF, "indic"), (0x19FF, "khmer_myanmar"), (0x1A1F, "indic"),
        (0x1AAF, "indic"), (0x1B7F, "indic"), (0x1BBF, "indic"), (0x1BFF, "indic"),
        (0x1C4F, "indic"), (0x1C7F, "indic"), (0x1C8F, "cyrillic"), (0x1CBF, "georgian"),
        (0x1CCF, "indic"), (0x1CFF, "indic"), (0x1D7F, "latin"), (0x1DBF, "latin"),
        (0x1DFF, "default"), (0x1EFF, "latin"), (0x309F, "kana"), (0x30FF, "kana"),
        (0x312F, "cjk"), (0x318F, "hangul"), (0x9FFF, "cjk"), (0xA4CF, "yi"),
        (0xA4FF, "default"), (0xA63F, "default"), (0xA69F, "cyrillic"), (0xA6FF, "default"),
        (0xA7FF, "latin"), (0xA82F, "indic"), (0xA87F, "default"), (0xA8DF, "indic"),
        (0xA8FF, "indic"), (0xA92F, "indic"), (0xA95F, "indic"), (0xA97F, "hangul"),
        (0xA9DF, "indic"), (0xA9FF, "khmer_myanmar"), (0xAA5F, "indic"), (0xAA7F, "khmer_myanmar"),
        (0xAADF, "indic"), (0xAAFF, "indic"), (0xAB2F, "ethiopic"), (0xAB6F, "latin"),
        (0xABBF, "default"), (0xABFF, "indic"), (0xD7AF, "hangul"), (0xFAFF, "cjk"),
        (0xFDFF, "arabic"), (0xFE6F, "default"), (0xFEFF, "arabic"), (0xFFEF, "latin"),
    ]  # fmt: skip

    breakpoints = [block_end for block_end, _ in ranges]

    @lru_cache(maxsize=4096)
    def _character_weight(self, character: str) -> float:
        code = ord(character)
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]
        if code == 32:
            return self.weights["space"]
        # Arabic Tatweel only stretches the glyph, it is not pronounced.
        if code == 0x0640:
            return self.weights["mark"]

        category = unicodedata.category(character)
        if category.startswith("M"):
            return self.weights["mark"]
        if category.startswith(("P", "S")):
            return self.weights["punctuation"]
        if category.startswith("Z"):
            return self.weights["space"]
        if category.startswith("N"):
            return self.weights["digit"]

        index = bisect.bisect_left(self.breakpoints, code)
        if index < len(self.ranges):
            return self.weights.get(self.ranges[index][1], self.weights["default"])
        if code > 0x20000:
            return self.weights["cjk"]
        return self.weights["default"]

    def total_weight(self, text: str) -> float:
        """Sums the per-character weights of `text`."""
        return sum(self._character_weight(character) for character in text)

    def estimate_duration(
        self,
        target_text: str,
        reference_text: str,
        reference_duration: float,
        low_threshold: float | None = 50,
        boost_strength: float = 3,
    ) -> float:
        """
        Estimates the duration of `target_text` from the speaking rate implied by a reference pair.

        Args:
            target_text (`str`):
                Text whose duration is estimated.
            reference_text (`str`):
                Transcript of the reference audio.
            reference_duration (`float`):
                Duration of the reference audio, in the same unit the estimate is returned in.
            low_threshold (`float`, *optional*, defaults to 50):
                Estimates below this value are boosted along a power curve, because short utterances are spoken
                proportionally slower than long ones.
            boost_strength (`float`, *optional*, defaults to 3):
                Strength of that boost. `1` disables it, `2` makes it square-root shaped.

        Returns:
            `float`: The estimated duration.
        """
        if reference_duration <= 0 or not reference_text:
            return 0.0
        reference_weight = self.total_weight(reference_text)
        if reference_weight == 0:
            return 0.0

        estimated = self.total_weight(target_text) / (reference_weight / reference_duration)
        if low_threshold is not None and estimated < low_threshold:
            return low_threshold * (estimated / low_threshold) ** (1.0 / boost_strength)
        return estimated


@requires(backends=("torch",))
class OmniVoiceProcessor(ProcessorMixin):
    r"""
    Constructs an OmniVoice processor which wraps a [`DacFeatureExtractor`], an [`AutoTokenizer`] and a
    [`HiggsAudioV2TokenizerModel`] into a single processor. See [`~OmniVoiceProcessor.__call__`] and
    [`~OmniVoiceProcessor.decode`] for more information.

    Args:
        feature_extractor ([`DacFeatureExtractor`]):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`AutoTokenizer`]):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer ([`HiggsAudioV2TokenizerModel`]):
            An instance of [`HiggsAudioV2TokenizerModel`]. The audio tokenizer is a required input.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of residual codebooks the audio tokenizer produces for one audio frame.
        audio_mask_id (`int`, *optional*, defaults to 1024):
            Id, within a codebook's vocabulary, of the mask token that marks a frame as not yet decoded.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, of the waveforms the audio tokenizer consumes and produces.
        frame_rate (`int`, *optional*, defaults to 25):
            Number of audio frames the audio tokenizer produces per second.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioV2TokenizerModel"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
        num_codebooks: int = 8,
        audio_mask_id: int = 1024,
        sampling_rate: int = 24_000,
        frame_rate: int = 25,
    ):
        self.num_codebooks = num_codebooks
        self.audio_mask_id = audio_mask_id
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.duration_estimator = OmniVoiceDurationEstimator()

        if feature_extractor is not None and audio_tokenizer is not None:
            super().__init__(
                feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template
            )
        else:
            # `ProcessorMixin.__init__` requires every declared attribute to be a real instance of the matching
            # class, so it cannot build a tokenizer-only processor. Wire that case up by hand instead.
            self.feature_extractor = feature_extractor
            self.audio_tokenizer = audio_tokenizer
            self.tokenizer = tokenizer
            self.chat_template = chat_template

        if audio_tokenizer is not None:
            self.sampling_rate = audio_tokenizer.config.sample_rate
            self.frame_rate = audio_tokenizer.config.frame_rate

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, audio_tokenizer_id: str = DEFAULT_AUDIO_TOKENIZER_ID, **kwargs
    ):
        r"""
        Loads the text tokenizer from the checkpoint root and the audio tokenizer from its `audio_tokenizer`
        subfolder, which is how OmniVoice checkpoints are laid out.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Repository id or local path of an OmniVoice checkpoint.
            audio_tokenizer_id (`str`, *optional*, defaults to `"eustlb/higgs-audio-v2-tokenizer"`):
                Repository id of the [`HiggsAudioV2TokenizerModel`] to fall back to when the checkpoint does not
                bundle its own `audio_tokenizer` subfolder.
            kwargs:
                Forwarded to every component's `from_pretrained`.

        Returns:
            [`OmniVoiceProcessor`]: The loaded processor.
        """
        from transformers import AutoTokenizer, DacFeatureExtractor, HiggsAudioV2TokenizerModel

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        audio_tokenizer_source = pretrained_model_name_or_path
        audio_tokenizer_kwargs = {"subfolder": "audio_tokenizer", **kwargs}
        try:
            feature_extractor = DacFeatureExtractor.from_pretrained(
                audio_tokenizer_source, **audio_tokenizer_kwargs
            )
        except OSError:
            logger.warning_once(
                f"'{pretrained_model_name_or_path}' does not bundle an `audio_tokenizer` subfolder; falling back "
                f"to '{audio_tokenizer_id}'."
            )
            audio_tokenizer_source = audio_tokenizer_id
            audio_tokenizer_kwargs = dict(kwargs)
            feature_extractor = DacFeatureExtractor.from_pretrained(
                audio_tokenizer_source, **audio_tokenizer_kwargs
            )

        audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_source, **audio_tokenizer_kwargs
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, audio_tokenizer=audio_tokenizer)

    @property
    def supported_language_ids(self) -> frozenset[str]:
        """The language ids accepted by the `language` argument of [`~OmniVoiceProcessor.__call__`]."""
        return LANG_IDS

    @property
    def supported_language_names(self) -> frozenset[str]:
        """The language names accepted by the `language` argument of [`~OmniVoiceProcessor.__call__`]."""
        return LANG_NAMES

    def chunk_text(self, text: str, chunk_duration: float = 15.0, reference_text: str | None = None) -> list[str]:
        """
        Splits `text` at punctuation into pieces that synthesize to roughly `chunk_duration` seconds each.

        Args:
            text (`str`):
                Text to split.
            chunk_duration (`float`, *optional*, defaults to 15.0):
                Target duration, in seconds, of the audio each chunk synthesizes to.
            reference_text (`str`, *optional*):
                Transcript of the reference audio, used to calibrate the speaking rate.

        Returns:
            `list[str]`: The text chunks, in order.
        """
        target_frames = self.estimate_target_length(text, reference_text=reference_text)
        characters_per_chunk = int(chunk_duration * self.frame_rate * len(text) / max(target_frames, 1))
        return _chunk_text_punctuation(text, chunk_len=max(characters_per_chunk, 1), min_chunk_len=3)

    def estimate_target_length(
        self,
        text: str,
        reference_text: str | None = None,
        reference_length: int | None = None,
        speed: float = 1.0,
    ) -> int:
        """
        Estimates how many audio frames `text` needs.

        Args:
            text (`str`):
                Text to synthesize.
            reference_text (`str`, *optional*):
                Transcript of the reference audio.
            reference_length (`int`, *optional*):
                Number of audio frames of the reference audio.
            speed (`float`, *optional*, defaults to 1.0):
                Speaking rate factor. Values above `1.0` shorten the estimate.

        Returns:
            `int`: The number of audio frames to generate.
        """
        if reference_length is None or not reference_text:
            # Calibrate against a short English utterance when there is no reference pair to measure.
            reference_text, reference_length = "Nice to meet you.", 25
        estimate = self.duration_estimator.estimate_duration(text, reference_text, reference_length)
        if speed > 0 and speed != 1.0:
            estimate = estimate / speed
        return max(1, int(estimate))

    def encode_audio(self, audio, sampling_rate: int | None = None) -> torch.LongTensor:
        """
        Encodes a waveform into audio codes with the bundled [`HiggsAudioV2TokenizerModel`].

        Args:
            audio (`np.ndarray` or `torch.Tensor` of shape `(num_samples,)`, `(1, num_samples)` or `(channels, num_samples)`):
                Waveform to encode. Multi-channel input is averaged down to mono.
            sampling_rate (`int`, *optional*):
                Sampling rate of `audio`. Defaults to the processor's own sampling rate; anything else is
                resampled.

        Returns:
            `torch.LongTensor` of shape `(num_codebooks, num_frames)`: The audio codes.
        """
        if self.audio_tokenizer is None:
            raise ValueError(
                "This processor was loaded without an audio tokenizer, so it cannot encode reference audio."
            )

        waveform = self._to_mono_tensor(audio, sampling_rate)

        duration = waveform.shape[-1] / self.sampling_rate
        if duration > 20.0:
            logger.warning(
                f"The reference audio is {duration:.1f}s long. Anything above 20s slows generation down, raises "
                f"memory use and degrades cloning quality; 3-10s works best."
            )

        # A quiet reference is boosted before encoding; `decode` scales the generated waveform back by the same
        # factor when it is given the reference.
        rms = float(waveform.pow(2).mean().sqrt())
        if 0 < rms < 0.1:
            waveform = waveform * 0.1 / rms

        # The codec consumes whole frames, so the trailing partial frame is dropped.
        remainder = waveform.shape[-1] % self.audio_tokenizer.config.hop_length
        if remainder:
            waveform = waveform[..., :-remainder]

        waveform = waveform.to(self.audio_tokenizer.device, dtype=self.audio_tokenizer.dtype)
        with torch.no_grad():
            return self.audio_tokenizer.encode(waveform.unsqueeze(0)).audio_codes.squeeze(0)

    def _to_mono_tensor(self, audio, sampling_rate: int | None) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio)
        elif isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = torch.tensor(audio)
        waveform = waveform.to(torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sampling_rate, new_freq=self.sampling_rate
            )
        return waveform

    def _build_style_ids(
        self, language: str | None, instruct: str | None, denoise: bool
    ) -> torch.Tensor:
        style = "<|denoise|>" if denoise else ""
        style += f"<|lang_start|>{language if language else 'None'}<|lang_end|>"
        style += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"
        return self.tokenizer(style, return_tensors="pt").input_ids.repeat(self.num_codebooks, 1)

    def __call__(
        self,
        text: TextInput | list[TextInput],
        language: str | list[str] | None = None,
        instruct: str | list[str] | None = None,
        reference_text: str | list[str] | None = None,
        reference_audio=None,
        reference_audio_codes: torch.Tensor | list[torch.Tensor] | None = None,
        audio_codes: torch.Tensor | list[torch.Tensor] | None = None,
        sampling_rate: int | None = None,
        duration: float | list[float] | None = None,
        speed: float | list[float] | None = None,
        denoise: bool = True,
        add_reference_punctuation: bool = True,
        output_labels: bool = False,
        prompt_ratio: float | tuple[float, float] = 0.0,
        mask_ratio: float | tuple[float, float] = 1.0,
        drop_conditioning: bool = False,
    ) -> BatchFeature:
        r"""
        Prepares one or more prompts for [`OmniVoiceForConditionalGeneration`].

        Without `audio_codes` the processor builds a generation prompt: style markers, the text, the optional
        reference-audio codes, and a fully masked canvas of the estimated length. With `audio_codes` it builds a
        training example instead, masking part of the supplied codes and, when `output_labels` is set, emitting
        the labels of the masked positions.

        Args:
            text (`str` or `list[str]`):
                Text to synthesize.
            language (`str` or `list[str]`, *optional*):
                Language id (e.g. `"en"`) or name (e.g. `"English"`). `None` selects language-agnostic mode.
            instruct (`str` or `list[str]`, *optional*):
                Voice-design attributes, e.g. `"female, low pitch, british accent"`. See
                [`~OmniVoiceProcessor.__call__`] validation rules in `_resolve_instruct`.
            reference_text (`str` or `list[str]`, *optional*):
                Transcript of the reference audio, required for voice cloning.
            reference_audio (`np.ndarray` or `torch.Tensor` or `list`, *optional*):
                Reference waveform to clone. Encoded with [`~OmniVoiceProcessor.encode_audio`]. Ignored when
                `reference_audio_codes` is given.
            reference_audio_codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)` or `list`, *optional*):
                Already encoded reference audio, as returned by [`~OmniVoiceProcessor.encode_audio`].
            audio_codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)` or `list`, *optional*):
                Codes of the target audio. Supplying them switches the processor to training mode.
            sampling_rate (`int`, *optional*):
                Sampling rate of `reference_audio`.
            duration (`float` or `list[float]`, *optional*):
                Fixed output duration in seconds, overriding the estimate derived from the text.
            speed (`float` or `list[float]`, *optional*):
                Speaking rate factor applied to the estimated duration. Ignored where `duration` is set.
            denoise (`bool`, *optional*, defaults to `True`):
                Whether to prepend the `<|denoise|>` marker. It only applies to items that have reference audio.
            add_reference_punctuation (`bool`, *optional*, defaults to `True`):
                Whether to append end punctuation to `reference_text` when it has none, so that it joins the
                target text as a separate sentence.
            output_labels (`bool`, *optional*, defaults to `False`):
                Whether to return `labels` for the masked audio frames. Requires `audio_codes`.
            prompt_ratio (`float` or `tuple[float, float]`, *optional*, defaults to 0.0):
                Fraction of the leading audio frames kept unmasked and excluded from the loss. A tuple is a range
                sampled uniformly per item.
            mask_ratio (`float` or `tuple[float, float]`, *optional*, defaults to 1.0):
                Probability of masking each codebook entry outside the prompt region. A tuple is a range sampled
                uniformly per item.
            drop_conditioning (`bool`, *optional*, defaults to `False`):
                Whether to drop the style and text prefix entirely, which trains the unconditional branch used by
                classifier-free guidance.

        Returns:
            [`BatchFeature`]: `input_ids` of shape `(batch_size, num_codebooks, sequence_length)`, `audio_mask`
            and `attention_mask` of shape `(batch_size, sequence_length)`, plus `target_lengths` of shape
            `(batch_size,)` in generation mode, or `labels` of shape
            `(batch_size, num_codebooks, sequence_length)` in training mode when `output_labels` is set.
        """
        texts = [text] if isinstance(text, str) else list(text)
        batch_size = len(texts)

        languages = self._broadcast(language, batch_size)
        instructs = self._broadcast(instruct, batch_size)
        reference_texts = self._broadcast(reference_text, batch_size)
        durations = self._broadcast(duration, batch_size)
        speeds = self._broadcast(speed, batch_size)
        target_codes = self._broadcast_tensor(audio_codes, batch_size)

        if reference_audio_codes is not None:
            reference_codes = self._broadcast_tensor(reference_audio_codes, batch_size)
        elif reference_audio is not None:
            reference_audios = reference_audio if isinstance(reference_audio, list) else [reference_audio]
            reference_codes = [self.encode_audio(item, sampling_rate) for item in reference_audios]
            reference_codes = self._broadcast_tensor(reference_codes, batch_size)
        else:
            reference_codes = [None] * batch_size

        languages = [_resolve_language(item) for item in languages]
        instructs = [
            _resolve_instruct(item, use_zh=bool(texts[i] and _ZH_RE.search(texts[i])))
            for i, item in enumerate(instructs)
        ]
        if add_reference_punctuation:
            reference_texts = [_add_punctuation(item) if item else item for item in reference_texts]

        rows = []
        for i in range(batch_size):
            if target_codes[i] is None:
                rows.append(
                    self._build_generation_row(
                        text=texts[i],
                        language=languages[i],
                        instruct=instructs[i],
                        reference_text=reference_texts[i],
                        reference_codes=reference_codes[i],
                        duration=durations[i],
                        speed=speeds[i],
                        denoise=denoise,
                    )
                )
            else:
                rows.append(
                    self._build_training_row(
                        text=texts[i],
                        language=languages[i],
                        instruct=instructs[i],
                        audio_codes=target_codes[i],
                        prompt_ratio=prompt_ratio,
                        mask_ratio=mask_ratio,
                        drop_conditioning=drop_conditioning,
                    )
                )

        return self._collate(rows, output_labels=output_labels)

    def _build_generation_row(
        self,
        text: str,
        language: str | None,
        instruct: str | None,
        reference_text: str | None,
        reference_codes: torch.Tensor | None,
        duration: float | None,
        speed: float | None,
        denoise: bool,
    ) -> dict:
        style_ids = self._build_style_ids(language, instruct, denoise and reference_codes is not None)
        combined_text = _combine_text(text, reference_text)
        text_ids = _tokenize_with_nonverbal_tags(f"<|text_start|>{combined_text}<|text_end|>", self.tokenizer)
        text_ids = text_ids.repeat(self.num_codebooks, 1)

        if duration is not None:
            target_length = max(1, int(duration * self.frame_rate))
        else:
            target_length = self.estimate_target_length(
                text,
                reference_text=reference_text,
                reference_length=reference_codes.shape[-1] if reference_codes is not None else None,
                speed=speed if speed is not None else 1.0,
            )
        canvas = torch.full((self.num_codebooks, target_length), self.audio_mask_id, dtype=torch.long)

        parts = [style_ids, text_ids]
        if reference_codes is not None:
            parts.append(reference_codes.to(torch.long).cpu())
        parts.append(canvas)
        input_ids = torch.cat(parts, dim=1)

        audio_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        audio_mask[style_ids.shape[1] + text_ids.shape[1] :] = True

        return {"input_ids": input_ids, "audio_mask": audio_mask, "target_length": target_length}

    def _build_training_row(
        self,
        text: str,
        language: str | None,
        instruct: str | None,
        audio_codes: torch.Tensor,
        prompt_ratio: float | tuple[float, float],
        mask_ratio: float | tuple[float, float],
        drop_conditioning: bool,
    ) -> dict:
        audio_codes = audio_codes.to(torch.long).cpu()
        prompt_ratio = 0.0 if drop_conditioning else self._sample_ratio(prompt_ratio)
        sampled_mask_ratio = self._sample_ratio(mask_ratio)

        prompt_length = int(audio_codes.shape[1] * prompt_ratio)
        audio_inputs = audio_codes.clone()
        audio_labels = audio_codes.clone()

        masked = torch.rand(audio_codes[:, prompt_length:].shape) < sampled_mask_ratio
        audio_inputs[:, prompt_length:][masked] = self.audio_mask_id
        # The loss is only defined on the entries the model actually had to predict.
        audio_labels[:, prompt_length:][~masked] = -100
        if not drop_conditioning:
            audio_labels[:, :prompt_length] = -100

        if drop_conditioning:
            return {
                "input_ids": audio_inputs,
                "labels": audio_labels,
                "audio_mask": torch.ones(audio_inputs.shape[1], dtype=torch.bool),
                "target_length": audio_inputs.shape[1] - prompt_length,
            }

        style_ids = self._build_style_ids(language, instruct, denoise=False)
        text_ids = self.tokenizer(f"<|text_start|>{text}<|text_end|>", return_tensors="pt").input_ids.repeat(
            self.num_codebooks, 1
        )
        prefix_length = style_ids.shape[1] + text_ids.shape[1]

        input_ids = torch.cat([style_ids, text_ids, audio_inputs], dim=1)
        labels = torch.cat(
            [torch.full(style_ids.shape, -100), torch.full(text_ids.shape, -100), audio_labels], dim=1
        )
        audio_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        audio_mask[prefix_length:] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "target_length": audio_inputs.shape[1] - prompt_length,
        }

    def _collate(self, rows: list[dict], output_labels: bool) -> BatchFeature:
        batch_size = len(rows)
        max_length = max(row["input_ids"].shape[1] for row in rows)
        has_labels = all("labels" in row for row in rows)
        if output_labels and not has_labels:
            raise ValueError("`output_labels=True` requires `audio_codes` to be passed for every item.")

        input_ids = torch.full((batch_size, self.num_codebooks, max_length), self.audio_mask_id, dtype=torch.long)
        audio_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, self.num_codebooks, max_length), -100, dtype=torch.long)

        for i, row in enumerate(rows):
            length = row["input_ids"].shape[1]
            input_ids[i, :, :length] = row["input_ids"]
            audio_mask[i, :length] = row["audio_mask"]
            attention_mask[i, :length] = True
            position_ids[i, :length] = torch.arange(length)
            if has_labels:
                labels[i, :, :length] = row["labels"]

        data = {"input_ids": input_ids, "audio_mask": audio_mask, "attention_mask": attention_mask}
        if output_labels:
            data["labels"] = labels
            data["position_ids"] = position_ids
        else:
            data["target_lengths"] = torch.tensor([row["target_length"] for row in rows], dtype=torch.long)
        return BatchFeature(data=data, tensor_type="pt")

    def decode(
        self,
        audio_codes: torch.Tensor | list[torch.Tensor],
        reference_audio=None,
        reference_rms: float | None = None,
        sampling_rate: int | None = None,
        pad_duration: float = 0.1,
        fade_duration: float = 0.1,
    ) -> np.ndarray:
        """
        Decodes one item's audio codes back into a waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)` or `list`):
                Codes to decode. A list is treated as consecutive chunks of one utterance and cross-faded
                together.
            reference_audio (`np.ndarray` or `torch.Tensor`, *optional*):
                Reference waveform the voice was cloned from, used to match its loudness.
            reference_rms (`float`, *optional*):
                Root mean square of that reference waveform, if already known. Takes precedence over
                `reference_audio`. When neither is given the waveform is peak-normalized instead.
            sampling_rate (`int`, *optional*):
                Sampling rate of `reference_audio`.
            pad_duration (`float`, *optional*, defaults to 0.1):
                Silence added at each edge, in seconds. `0` disables it.
            fade_duration (`float`, *optional*, defaults to 0.1):
                Length of the fade-in and fade-out curves, in seconds. `0` disables them.

        Returns:
            `np.ndarray` of shape `(num_samples,)`: The waveform, at [`~OmniVoiceProcessor.sampling_rate`] Hz.
        """
        if self.audio_tokenizer is None:
            raise ValueError("This processor was loaded without an audio tokenizer, so it cannot decode codes.")

        chunks = audio_codes if isinstance(audio_codes, list) else [audio_codes]
        waveforms = []
        for chunk in chunks:
            chunk = self._trim_masked_frames(chunk).to(self.audio_tokenizer.device)
            with torch.no_grad():
                decoded = self.audio_tokenizer.decode(chunk.unsqueeze(0)).audio_values[0]
            waveforms.append(decoded.float().cpu().numpy())
        waveform = _cross_fade_chunks(waveforms, self.sampling_rate)

        if reference_rms is None and reference_audio is not None:
            reference_rms = float(self._to_mono_tensor(reference_audio, sampling_rate).pow(2).mean().sqrt())
        if reference_rms is not None:
            # `__call__` boosts a quiet reference to an RMS of 0.1 before encoding it, so the output is scaled
            # back by the same factor here.
            if reference_rms < 0.1:
                waveform = waveform * reference_rms / 0.1
        else:
            peak = np.abs(waveform).max()
            if peak > 1e-6:
                waveform = waveform / peak * 0.5

        waveform = _fade_and_pad_audio(waveform, pad_duration, fade_duration, self.sampling_rate)
        return waveform.squeeze(0)

    def batch_decode(self, audio_codes: torch.Tensor, **kwargs) -> list[np.ndarray]:
        """
        Decodes a batch of audio codes back into waveforms.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                Codes as returned by [`~OmniVoiceForConditionalGeneration.generate`].
            kwargs:
                Forwarded to [`~OmniVoiceProcessor.decode`].

        Returns:
            `list[np.ndarray]`: One waveform per item.
        """
        return [self.decode(item, **kwargs) for item in audio_codes]

    def _trim_masked_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Drops the trailing frames `generate` left masked because they belong to a shorter batch item."""
        generated = (audio_codes != self.audio_mask_id).any(dim=0)
        if not bool(generated.any()):
            raise ValueError("The audio codes hold no generated frame.")
        return audio_codes[:, : int(generated.nonzero()[-1]) + 1]

    @staticmethod
    def _sample_ratio(ratio: float | tuple[float, float]) -> float:
        if isinstance(ratio, (tuple, list)):
            return random.uniform(*ratio)
        return float(ratio)

    @staticmethod
    def _broadcast(value, batch_size: int) -> list:
        if not isinstance(value, list):
            value = [value]
        if len(value) == 1:
            value = value * batch_size
        if len(value) != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} values, but got {len(value)}.")
        return list(value)

    @staticmethod
    def _broadcast_tensor(value, batch_size: int) -> list:
        if value is None:
            return [None] * batch_size
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            value = list(value)
        elif not isinstance(value, list):
            value = [value]
        if len(value) == 1:
            value = value * batch_size
        if len(value) != batch_size:
            raise ValueError(f"Expected 1 or {batch_size} values, but got {len(value)}.")
        return list(value)


__all__ = ["OmniVoiceDurationEstimator", "OmniVoiceProcessor"]
